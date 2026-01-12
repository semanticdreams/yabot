import asyncio
import base64
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from nacl.signing import SigningKey

from nio import Api, AsyncClient
from nio.responses import Response

from .storage import atomic_write_json, load_json


@dataclass(frozen=True)
class SigningKeyPair:
    public: str
    private: str

    @property
    def key_id(self) -> str:
        return f"ed25519:{self.public}"

    def signing_key(self) -> SigningKey:
        seed = _b64decode(self.private)
        return SigningKey(seed)


@dataclass
class CrossSigningKeys:
    master: SigningKeyPair
    self_signing: SigningKeyPair
    user_signing: SigningKeyPair


@dataclass
class RawResponse(Response):
    data: Dict[str, Any]

    @classmethod
    def from_dict(cls, parsed_dict: Dict[str, Any], *_args, **_kwargs) -> "RawResponse":
        return cls(parsed_dict)


def _b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii").rstrip("=")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.b64decode(data + padding)


class CrossSigningManager:
    def __init__(self, data_dir: str, reset: bool = False, password: Optional[str] = None) -> None:
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "cross_signing.json")
        self.uia_path = os.path.join(data_dir, "cross_signing_uia.json")
        self.logger = logging.getLogger("matrix_bot.cross_signing")
        self._lock = asyncio.Lock()
        self._keys: Optional[CrossSigningKeys] = None
        self._server_ready = False
        self._disabled = False
        self._reset = reset
        self._reset_done = False
        self._mismatch_retry = False
        self._password = password
        self._uia_session: Optional[str] = None

    async def ensure_setup(self, client: AsyncClient) -> None:
        async with self._lock:
            if self._disabled:
                return
            self._load_uia_session()
            master_keys = await self._query_master_keys(client)
            if master_keys is None:
                self.logger.warning("Skipping cross-signing setup: unable to query keys")
                return
            if self._reset and not self._reset_done:
                stored = load_json(self.path)
                if isinstance(stored, dict) and "master" in stored:
                    try:
                        self._keys = CrossSigningKeys(
                            master=SigningKeyPair(**stored["master"]),
                            self_signing=SigningKeyPair(**stored["self_signing"]),
                            user_signing=SigningKeyPair(**stored["user_signing"]),
                        )
                    except Exception:
                        self.logger.warning("Invalid local cross-signing keys; regenerating")
                        self._keys = None
                if not self._keys:
                    self._keys = self._generate_keys()
                    atomic_write_json(
                        self.path,
                        {
                            "master": {"public": self._keys.master.public, "private": self._keys.master.private},
                            "self_signing": {
                                "public": self._keys.self_signing.public,
                                "private": self._keys.self_signing.private,
                            },
                            "user_signing": {
                                "public": self._keys.user_signing.public,
                                "private": self._keys.user_signing.private,
                            },
                        },
                    )
                    self.logger.warning("Cross-signing reset enabled; generated new keys")
                self._reset_done = True
            else:
                await self._ensure_keys(client, master_keys)
            if self._disabled:
                return
            await self._ensure_server_keys(client, master_keys, force_reset=self._reset)
            if self._disabled:
                return
            await self._sign_own_devices(client)

    async def _ensure_keys(self, client: AsyncClient, master_keys: Optional[Dict[str, Any]]) -> None:
        if self._keys:
            return
        stored = load_json(self.path)
        if isinstance(stored, dict) and "master" in stored:
            try:
                self._keys = CrossSigningKeys(
                    master=SigningKeyPair(**stored["master"]),
                    self_signing=SigningKeyPair(**stored["self_signing"]),
                    user_signing=SigningKeyPair(**stored["user_signing"]),
                )
                return
            except Exception:
                self.logger.warning("Invalid cross-signing key file, regenerating")

        user_id = client.user_id
        if master_keys and user_id and user_id in master_keys:
            self.logger.error(
                "Cross-signing keys exist on server but no local key file; refusing to overwrite"
            )
            self._disabled = True
            return

        self._keys = self._generate_keys()
        atomic_write_json(
            self.path,
            {
                "master": {"public": self._keys.master.public, "private": self._keys.master.private},
                "self_signing": {
                    "public": self._keys.self_signing.public,
                    "private": self._keys.self_signing.private,
                },
                "user_signing": {
                    "public": self._keys.user_signing.public,
                    "private": self._keys.user_signing.private,
                },
            },
        )
        self.logger.info("Generated new cross-signing keys")

    async def _ensure_server_keys(
        self,
        client: AsyncClient,
        master_keys: Optional[Dict[str, Any]],
        force_reset: bool = False,
    ) -> None:
        if self._server_ready:
            return
        if not self._keys:
            return
        if master_keys is None:
            return

        user_id = client.user_id
        if not user_id:
            return
        if not client.olm:
            self.logger.warning("E2EE not enabled; skipping cross-signing setup")
            return

        if master_keys and user_id in master_keys and not force_reset:
            server_keys = master_keys[user_id].get("keys", {})
            if self._keys.master.key_id not in server_keys:
                self.logger.error(
                    "Server master key does not match local key file local=%s server=%s",
                    self._keys.master.key_id,
                    ",".join(server_keys.keys()),
                )
                if not self._mismatch_retry:
                    self._mismatch_retry = True
                    self.logger.warning("Retrying cross-signing upload due to key mismatch")
                else:
                    self._disabled = True
                    return
            else:
                self._server_ready = True
                return
        if master_keys and user_id in master_keys and force_reset:
            self.logger.warning("Force-resetting cross-signing keys on server")

        device_key_id, device_signature = self._sign_with_device_key(client, {
            "user_id": user_id,
            "usage": ["master"],
            "keys": {self._keys.master.key_id: self._keys.master.public},
        })

        master_key = {
            "user_id": user_id,
            "usage": ["master"],
            "keys": {self._keys.master.key_id: self._keys.master.public},
            "signatures": {user_id: {device_key_id: device_signature}},
        }

        self_signing_key = self._signed_key(
            user_id,
            ["self_signing"],
            self._keys.self_signing,
            self._keys.master,
        )
        user_signing_key = self._signed_key(
            user_id,
            ["user_signing"],
            self._keys.user_signing,
            self._keys.master,
        )

        content = {
            "master_key": master_key,
            "self_signing_key": self_signing_key,
            "user_signing_key": user_signing_key,
        }
        if force_reset and self._uia_session:
            content["auth"] = {
                "type": "org.matrix.cross_signing_reset",
                "session": self._uia_session,
            }
        elif force_reset and self._password and user_id:
            content["auth"] = {
                "type": "m.login.password",
                "user": user_id,
                "identifier": {"type": "m.id.user", "user": user_id},
                "password": self._password,
            }

        try:
            path = Api._build_path(["keys", "device_signing", "upload"], {"access_token": client.access_token})
            resp = await client._send(RawResponse, "POST", path, Api.to_json(content))  # type: ignore[attr-defined]
            if resp.data.get("session") and resp.data.get("flows"):
                self._store_uia_session(resp.data.get("session"))
                self.logger.warning("Cross-signing reset requires approval; follow server URL and retry")
                return
            if resp.data.get("errcode"):
                self.logger.error("Cross-signing upload failed errcode=%s", resp.data.get("errcode"))
                self._disabled = True
                return
        except Exception as exc:
            self.logger.exception("Cross-signing upload failed: %s", exc)
            return
        self.logger.info("Uploaded cross-signing keys")
        await asyncio.sleep(0.5)
        await self._verify_server_master_key(client)

    async def _sign_own_devices(self, client: AsyncClient) -> None:
        if not self._keys or not client.user_id or not self._server_ready:
            return
        if not client.olm:
            return

        try:
            method, path, data = Api.keys_query(client.access_token or "", {client.user_id})
            resp = await client._send(RawResponse, method, path, data)  # type: ignore[attr-defined]
        except Exception as exc:
            self.logger.exception("Failed to query device keys: %s", exc)
            return

        device_keys = resp.data.get("device_keys", {}).get(client.user_id, {})
        if not device_keys:
            return

        signed_payload: Dict[str, Dict[str, Any]] = {}
        for device_id, device in device_keys.items():
            signatures = device.get("signatures", {}).get(client.user_id, {})
            if self._keys.self_signing.key_id in signatures:
                continue

            signature = self._sign_with_key(self._keys.self_signing, device)
            device.setdefault("signatures", {}).setdefault(client.user_id, {})[self._keys.self_signing.key_id] = signature
            signed_payload[device_id] = device

        if not signed_payload:
            return

        content = {client.user_id: signed_payload}
        try:
            path = Api._build_path(["keys", "signatures", "upload"], {"access_token": client.access_token})
            resp = await client._send(RawResponse, "POST", path, Api.to_json(content))  # type: ignore[attr-defined]
            if resp.data.get("errcode"):
                self.logger.error("Signature upload failed errcode=%s", resp.data.get("errcode"))
                return
        except Exception as exc:
            self.logger.exception("Signature upload failed: %s", exc)
            return

        self.logger.info("Signed %d device(s)", len(signed_payload))

    async def _query_master_keys(self, client: AsyncClient) -> Optional[Dict[str, Any]]:
        user_id = client.user_id
        if not user_id:
            return None
        try:
            method, path, data = Api.keys_query(client.access_token or "", {user_id})
            resp = await client._send(RawResponse, method, path, data)  # type: ignore[attr-defined]
        except Exception as exc:
            self.logger.exception("Failed to query keys: %s", exc)
            return None
        return resp.data.get("master_keys", {})

    async def _verify_server_master_key(self, client: AsyncClient) -> None:
        master_keys = await self._query_master_keys(client)
        if not master_keys or not client.user_id or not self._keys:
            return
        server_keys = master_keys.get(client.user_id, {}).get("keys", {})
        if self._keys.master.key_id in server_keys:
            self._server_ready = True
            self._mismatch_retry = False
            self._clear_uia_session()
            return
        self.logger.error(
            "Server master key mismatch after upload local=%s server=%s",
            self._keys.master.key_id,
            ",".join(server_keys.keys()),
        )
        self._disabled = True

    def _load_uia_session(self) -> None:
        if self._uia_session is not None:
            return
        data = load_json(self.uia_path)
        if isinstance(data, dict) and data.get("session"):
            self._uia_session = str(data["session"])

    def _store_uia_session(self, session: Optional[str]) -> None:
        if not session:
            return
        self._uia_session = str(session)
        atomic_write_json(self.uia_path, {"session": self._uia_session})

    def _clear_uia_session(self) -> None:
        self._uia_session = None
        try:
            os.remove(self.uia_path)
        except FileNotFoundError:
            pass

    def _sign_with_device_key(self, client: AsyncClient, payload: Dict[str, Any]) -> tuple[str, str]:
        canonical = Api.to_canonical_json(self._strip_signatures(payload))
        signature = client.olm.account.sign(canonical)  # type: ignore[union-attr]
        device_id = client.device_id or ""
        key_id = f"ed25519:{device_id}"
        return key_id, signature

    def _signed_key(
        self,
        user_id: str,
        usage: list[str],
        target: SigningKeyPair,
        signer: SigningKeyPair,
    ) -> Dict[str, Any]:
        payload = {
            "user_id": user_id,
            "usage": usage,
            "keys": {target.key_id: target.public},
        }
        signature = self._sign_with_key(signer, payload)
        payload["signatures"] = {user_id: {signer.key_id: signature}}
        return payload

    def _sign_with_key(self, signer: SigningKeyPair, payload: Dict[str, Any]) -> str:
        canonical = Api.to_canonical_json(self._strip_signatures(payload)).encode("utf-8")
        sig = signer.signing_key().sign(canonical).signature
        return _b64encode(sig)

    @staticmethod
    def _strip_signatures(payload: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(payload)
        data.pop("signatures", None)
        data.pop("unsigned", None)
        return data

    @staticmethod
    def _generate_keypair() -> SigningKeyPair:
        sk = SigningKey.generate()
        seed = sk.encode()
        pub = sk.verify_key.encode()
        return SigningKeyPair(
            public=_b64encode(pub),
            private=_b64encode(seed),
        )

    def _generate_keys(self) -> CrossSigningKeys:
        return CrossSigningKeys(
            master=self._generate_keypair(),
            self_signing=self._generate_keypair(),
            user_signing=self._generate_keypair(),
        )
