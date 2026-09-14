"""Tree-aware /history endpoints used by the patched stock webui DatabaseService."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient

import grimoire.config as config
import grimoire.entrypoint as entrypoint
from grimoire.history import HistoryStore


class HistoryTreeContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "tree.sqlite3")
        self._old_store = entrypoint.history_store
        entrypoint.history_store = HistoryStore(self.db_path)
        self._old_api = config.API_KEY
        self._old_admin = config.ADMIN_TOKEN
        config.API_KEY = "test-key"
        config.ADMIN_TOKEN = "test-key"
        self.client = TestClient(entrypoint.app)
        self.auth = {
            config.INTERNAL_AUTH_SUB_HEADER: "test-user",
            config.INTERNAL_AUTH_ROLE_HEADER: "administrator",
        }

    def tearDown(self):
        entrypoint.history_store = self._old_store
        config.API_KEY = self._old_api
        config.ADMIN_TOKEN = self._old_admin
        self.tmp.cleanup()

    def _create_conv(self, conv_id="c1", name="Demo", curr_node=None):
        body = {"id": conv_id, "name": name, "lastModified": 1}
        if curr_node is not None:
            body["currNode"] = curr_node
        return self.client.post("/history", json=body, headers=self.auth).json()

    def _add_message(self, conv_id, msg_id, parent_id, role="user", content="hi", **extra):
        body = {"id": msg_id, "role": role, "content": content, "parent": parent_id}
        body.update(extra)
        return self.client.post(
            f"/history/{conv_id}/messages",
            json=body,
            headers=self.auth,
        ).json()

    @staticmethod
    def _auth_for(sub):
        return {
            config.INTERNAL_AUTH_SUB_HEADER: sub,
            config.INTERNAL_AUTH_ROLE_HEADER: "user",
        }

    def test_create_returns_webui_shape(self):
        conv = self._create_conv()
        self.assertEqual(conv["id"], "c1")
        self.assertEqual(conv["name"], "Demo")
        self.assertEqual(conv["messages"], [])
        self.assertIsNone(conv["currNode"])

    def test_list_orders_by_lastModified_desc(self):
        self._create_conv("a", "A")
        # Bump 'b' to a higher lastModified
        self.client.post("/history", json={"id": "b", "name": "B", "lastModified": 999}, headers=self.auth)
        listing = self.client.get("/history", headers=self.auth).json()["conversations"]
        self.assertEqual([c["id"] for c in listing], ["b", "a"])

    def test_create_message_branch_links_parent_and_updates_currNode(self):
        self._create_conv()
        root = self._add_message("c1", "m-root", parent_id=None, role="system", type="root", content="")
        self.assertEqual(root["parent"], None)
        self.assertEqual(root["children"], [])
        child = self._add_message("c1", "m-1", parent_id="m-root", role="user", content="hello")
        self.assertEqual(child["parent"], "m-root")
        # Refetching the conversation shows updated parent.children + currNode
        conv = self.client.get("/history/c1", headers=self.auth).json()
        self.assertEqual(conv["currNode"], "m-1")
        msgs_by_id = {m["id"]: m for m in conv["messages"]}
        self.assertEqual(msgs_by_id["m-root"]["children"], ["m-1"])

    def test_create_branch_with_unknown_parent_returns_404(self):
        self._create_conv()
        response = self.client.post(
            "/history/c1/messages",
            json={"id": "m-x", "parent": "ghost", "role": "user", "content": "x"},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 404)

    def test_patch_message_updates_content_and_tool_fields(self):
        self._create_conv()
        self._add_message("c1", "m-root", parent_id=None, role="system", type="root", content="")
        self._add_message("c1", "m-1", parent_id="m-root", role="assistant", content="first")
        response = self.client.patch(
            "/history/c1/messages/m-1",
            json={"content": "edited", "toolCalls": "[]", "model": "qwen-3.6-27B"},
            headers=self.auth,
        )
        # The endpoint switched from 204 to 200+JSON body (commit 7925382) to avoid
        # client parse errors on empty responses.
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["updated"], "m-1")
        conv = self.client.get("/history/c1", headers=self.auth).json()
        msg = next(m for m in conv["messages"] if m["id"] == "m-1")
        self.assertEqual(msg["content"], "edited")
        self.assertEqual(msg["toolCalls"], "[]")
        self.assertEqual(msg["model"], "qwen-3.6-27B")

    def test_patch_message_reparents_tree_links(self):
        self._create_conv()
        self._add_message("c1", "m-root", parent_id=None, role="system", type="root", content="")
        self._add_message("c1", "m-a", parent_id="m-root", role="user", content="a")
        self._add_message("c1", "m-b", parent_id="m-root", role="user", content="b")
        response = self.client.patch(
            "/history/c1/messages/m-b",
            json={"parent": "m-a"},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 200)
        conv = self.client.get("/history/c1", headers=self.auth).json()
        msgs_by_id = {m["id"]: m for m in conv["messages"]}
        self.assertEqual(msgs_by_id["m-root"]["children"], ["m-a"])
        self.assertEqual(msgs_by_id["m-a"]["children"], ["m-b"])

    def test_patch_message_rejects_parent_cycles(self):
        self._create_conv()
        self._add_message("c1", "m-root", parent_id=None)
        self._add_message("c1", "m-child", parent_id="m-root")

        self_parent = self.client.patch(
            "/history/c1/messages/m-root",
            json={"parent": "m-root"},
            headers=self.auth,
        )
        descendant_parent = self.client.patch(
            "/history/c1/messages/m-root",
            json={"parent": "m-child"},
            headers=self.auth,
        )

        self.assertEqual(self_parent.status_code, 400)
        self.assertEqual(descendant_parent.status_code, 400)

    def test_delete_message_unlinks_from_parent(self):
        self._create_conv()
        self._add_message("c1", "m-root", parent_id=None, role="system", type="root", content="")
        self._add_message("c1", "m-1", parent_id="m-root", role="user", content="a")
        self._add_message("c1", "m-2", parent_id="m-root", role="user", content="b")
        response = self.client.delete("/history/c1/messages/m-1", headers=self.auth)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["deleted"], ["m-1"])
        conv = self.client.get("/history/c1", headers=self.auth).json()
        ids = [m["id"] for m in conv["messages"]]
        self.assertNotIn("m-1", ids)
        root = next(m for m in conv["messages"] if m["id"] == "m-root")
        self.assertEqual(root["children"], ["m-2"])

    def test_delete_message_reparents_surviving_children(self):
        self._create_conv()
        self._add_message("c1", "m-root", parent_id=None)
        self._add_message("c1", "m-parent", parent_id="m-root")
        self._add_message("c1", "m-child", parent_id="m-parent")

        response = self.client.delete("/history/messages/m-parent", headers=self.auth)

        self.assertEqual(response.status_code, 200)
        conversation = self.client.get("/history/c1", headers=self.auth).json()
        messages = {message["id"]: message for message in conversation["messages"]}
        self.assertEqual(messages["m-root"]["children"], ["m-child"])
        self.assertEqual(messages["m-child"]["parent"], "m-root")

    def test_reparent_then_delete_keeps_server_managed_children_consistent(self):
        self._create_conv()
        self._add_message("c1", "m-root", parent_id=None, role="system", type="root", content="")
        self._add_message("c1", "m-system", parent_id="m-root", role="system", content="prompt")
        self._add_message("c1", "m-user", parent_id="m-system", role="user", content="hello")

        reparent = self.client.patch(
            "/history/messages/m-user", json={"parent": "m-root"}, headers=self.auth
        )
        deleted = self.client.delete("/history/messages/m-system", headers=self.auth)

        self.assertEqual(reparent.status_code, 200)
        self.assertEqual(deleted.status_code, 200)
        conversation = self.client.get("/history/c1", headers=self.auth).json()
        messages = {message["id"]: message for message in conversation["messages"]}
        self.assertEqual(set(messages), {"m-root", "m-user"})
        self.assertEqual(messages["m-root"]["children"], ["m-user"])
        self.assertEqual(messages["m-user"]["parent"], "m-root")

    def test_delete_current_node_clears_curr_node(self):
        self._create_conv()
        self._add_message("c1", "m-root", parent_id=None, role="system", type="root", content="")
        self._add_message("c1", "m-1", parent_id="m-root", role="user", content="a")
        conv = self.client.get("/history/c1", headers=self.auth).json()
        self.assertEqual(conv["currNode"], "m-1")
        response = self.client.delete("/history/c1/messages/m-1", headers=self.auth)
        self.assertEqual(response.status_code, 200)
        conv = self.client.get("/history/c1", headers=self.auth).json()
        self.assertIsNone(conv["currNode"])

    def test_delete_message_cascade_removes_subtree(self):
        self._create_conv()
        self._add_message("c1", "m-root", parent_id=None, role="system", type="root", content="")
        self._add_message("c1", "m-1", parent_id="m-root", role="user", content="a")
        self._add_message("c1", "m-1a", parent_id="m-1", role="assistant", content="a-reply")
        self._add_message("c1", "m-1b", parent_id="m-1", role="user", content="follow-up")
        response = self.client.delete("/history/c1/messages/m-1?cascade=true", headers=self.auth)
        self.assertEqual(response.status_code, 200)
        deleted = set(response.json()["deleted"])
        self.assertEqual(deleted, {"m-1", "m-1a", "m-1b"})
        conv = self.client.get("/history/c1", headers=self.auth).json()
        ids = {m["id"] for m in conv["messages"]}
        self.assertEqual(ids, {"m-root"})

    def test_fork_clones_path_and_links_origin(self):
        self._create_conv()
        self._add_message("c1", "m-root", parent_id=None, role="system", type="root", content="")
        self._add_message("c1", "m-1", parent_id="m-root", role="user", content="hi")
        self._add_message("c1", "m-2", parent_id="m-1", role="assistant", content="hello")
        response = self.client.post(
            "/history/c1/fork",
            json={"at_message_id": "m-2", "name": "Forked"},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 200)
        forked = response.json()
        self.assertEqual(forked["name"], "Forked")
        self.assertEqual(forked["forkedFromConversationId"], "c1")
        self.assertEqual(len(forked["messages"]), 3)
        self.assertEqual(forked["currNode"], forked["messages"][-1]["id"])
        # New IDs, not the originals
        new_ids = {m["id"] for m in forked["messages"]}
        self.assertFalse({"m-root", "m-1", "m-2"} & new_ids)

    def test_delete_with_forks_cascades_through_fork_chain(self):
        self._create_conv("parent-c")
        self.client.post("/history", json={"id": "child-c", "name": "Child", "forkedFromConversationId": "parent-c", "lastModified": 2}, headers=self.auth)
        self.client.post("/history", json={"id": "grand-c", "name": "Grand", "forkedFromConversationId": "child-c", "lastModified": 3}, headers=self.auth)
        response = self.client.delete("/history/parent-c?with_forks=true", headers=self.auth)
        # The endpoint switched from 204 to 200+JSON body (commit 7925382).
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["deleted"], "parent-c")
        listing = self.client.get("/history", headers=self.auth).json()["conversations"]
        self.assertEqual(listing, [])

    def test_delete_without_forks_reparents_children(self):
        self._create_conv("parent-c")
        self.client.post("/history", json={"id": "child-c", "name": "Child", "forkedFromConversationId": "parent-c", "lastModified": 2}, headers=self.auth)
        self.client.delete("/history/parent-c", headers=self.auth)
        survivor = self.client.get("/history/child-c", headers=self.auth).json()
        self.assertIsNone(survivor["forkedFromConversationId"])

    def test_patch_conversation_updates_curr_node_and_name(self):
        self._create_conv()
        self._add_message("c1", "m-root", parent_id=None, role="system", type="root", content="")
        response = self.client.patch(
            "/history/c1",
            json={"name": "Renamed", "currNode": "m-root"},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 200)
        conv = response.json()
        self.assertEqual(conv["name"], "Renamed")
        self.assertEqual(conv["currNode"], "m-root")

    def test_import_skips_duplicates(self):
        payload = [
            {
                "conv": {"id": "imp1", "name": "Imp", "lastModified": 1, "currNode": "m"},
                "messages": [
                    {"id": "m", "convId": "imp1", "role": "user", "content": "hi", "parent": None, "children": [], "type": "user", "timestamp": 1},
                ],
            }
        ]
        first = self.client.post("/history/import", json=payload, headers=self.auth).json()
        self.assertEqual(first, {"imported": 1, "skipped": 0})
        second = self.client.post("/history/import", json=payload, headers=self.auth).json()
        self.assertEqual(second, {"imported": 0, "skipped": 1})
        listing = self.client.get("/history", headers=self.auth).json()["conversations"]
        self.assertEqual(len(listing), 1)

    def test_reparent_cannot_link_to_another_accounts_message(self):
        alice = self._auth_for("alice")
        bob = self._auth_for("bob")
        self.client.post("/history", json={"id": "alice-c", "name": "Alice"}, headers=alice)
        self.client.post(
            "/history/alice-c/messages",
            json={"id": "alice-message", "role": "user", "content": "private"},
            headers=alice,
        )
        self.client.post("/history", json={"id": "bob-c", "name": "Bob"}, headers=bob)
        self.client.post(
            "/history/bob-c/messages",
            json={"id": "bob-message", "role": "user", "content": "hello"},
            headers=bob,
        )

        response = self.client.patch(
            "/history/bob-c/messages/bob-message",
            json={"parent": "alice-message"},
            headers=bob,
        )

        self.assertEqual(response.status_code, 404)
        alice_conversation = self.client.get("/history/alice-c", headers=alice).json()
        self.assertEqual(alice_conversation["messages"][0]["children"], [])

    def test_import_remaps_message_ids_owned_by_another_account(self):
        alice = self._auth_for("alice")
        bob = self._auth_for("bob")
        self.client.post("/history", json={"id": "alice-c", "name": "Alice"}, headers=alice)
        self.client.post(
            "/history/alice-c/messages",
            json={"id": "shared-id", "role": "user", "content": "alice-private"},
            headers=alice,
        )
        payload = [{
            "conv": {"id": "bob-import", "name": "Bob import", "currNode": "shared-id"},
            "messages": [{
                "id": "shared-id",
                "role": "user",
                "content": "bob-copy",
                "parent": None,
                "children": [],
            }],
        }]

        response = self.client.post("/history/import", json=payload, headers=bob)

        self.assertEqual(response.json(), {"imported": 1, "skipped": 0})
        alice_conversation = self.client.get("/history/alice-c", headers=alice).json()
        self.assertEqual(alice_conversation["messages"][0]["id"], "shared-id")
        self.assertEqual(alice_conversation["messages"][0]["content"], "alice-private")
        bob_conversation = self.client.get("/history/bob-import", headers=bob).json()
        self.assertNotEqual(bob_conversation["messages"][0]["id"], "shared-id")
        self.assertEqual(bob_conversation["currNode"], bob_conversation["messages"][0]["id"])

    def test_fork_rejects_dangling_parent_owned_by_another_account(self):
        alice = self._auth_for("alice")
        bob = self._auth_for("bob")
        self.client.post("/history", json={"id": "alice-c", "name": "Alice"}, headers=alice)
        self.client.post(
            "/history/alice-c/messages",
            json={"id": "alice-child", "role": "user", "content": "alice"},
            headers=alice,
        )
        self.client.post("/history", json={"id": "bob-c", "name": "Bob"}, headers=bob)
        self.client.post(
            "/history/bob-c/messages",
            json={"id": "foreign-parent", "role": "user", "content": "bob-private"},
            headers=bob,
        )
        with entrypoint.history_store._connect() as conn:
            conn.execute(
                "UPDATE messages SET parent_id = ? WHERE id = ?",
                ("foreign-parent", "alice-child"),
            )

        response = self.client.post(
            "/history/alice-c/fork",
            json={"at_message_id": "alice-child", "name": "Fork"},
            headers=alice,
        )

        self.assertEqual(response.status_code, 400)

    def test_import_skips_cyclic_message_tree(self):
        payload = [{
            "conv": {"id": "cyclic-import", "name": "Cyclic", "currNode": "a"},
            "messages": [
                {"id": "a", "parent": "b", "children": ["b"]},
                {"id": "b", "parent": "a", "children": ["a"]},
            ],
        }]

        response = self.client.post("/history/import", json=payload, headers=self.auth)

        self.assertEqual(response.json(), {"imported": 0, "skipped": 1})

    def test_import_skips_self_fork_and_cross_entry_fork_cycle(self):
        payload = [
            {"conv": {"id": "self", "name": "Self", "forkedFromConversationId": "self"}},
            {"conv": {"id": "fork-a", "name": "A", "forkedFromConversationId": "fork-b"}},
            {"conv": {"id": "fork-b", "name": "B", "forkedFromConversationId": "fork-a"}},
        ]

        response = self.client.post("/history/import", json=payload, headers=self.auth)

        self.assertEqual(response.json(), {"imported": 0, "skipped": 3})

    def test_import_detaches_fork_when_origin_is_unavailable(self):
        payload = [{
            "conv": {
                "id": "portable-fork",
                "name": "Portable",
                "forkedFromConversationId": "unavailable-origin",
                "currNode": "message",
            },
            "messages": [{
                "id": "message",
                "role": "user",
                "content": "portable",
                "parent": None,
                "children": [],
            }],
        }]

        response = self.client.post("/history/import", json=payload, headers=self.auth)

        self.assertEqual(response.json(), {"imported": 1, "skipped": 0})
        imported = self.client.get("/history/portable-fork", headers=self.auth).json()
        self.assertIsNone(imported["forkedFromConversationId"])

    def test_import_skips_children_that_disagree_with_parent_links(self):
        payload = [{
            "conv": {"id": "bad-children", "name": "Bad", "currNode": "root"},
            "messages": [{"id": "root", "parent": None, "children": ["root"]}],
        }]

        response = self.client.post("/history/import", json=payload, headers=self.auth)

        self.assertEqual(response.json(), {"imported": 0, "skipped": 1})

    def test_message_by_id_patch_returns_400_for_cycle(self):
        self._create_conv()
        self._add_message("c1", "m-root", parent_id=None)
        self._add_message("c1", "m-child", parent_id="m-root")

        response = self.client.patch(
            "/history/messages/m-root",
            json={"parent": "m-child"},
            headers=self.auth,
        )

        self.assertEqual(response.status_code, 400)

    def test_message_patch_rejects_children_that_disagree_with_parent_links(self):
        self._create_conv()
        self._add_message("c1", "m-root", parent_id=None)

        response = self.client.patch(
            "/history/messages/m-root",
            json={"children": ["m-root"]},
            headers=self.auth,
        )

        self.assertEqual(response.status_code, 400)

    def test_conversation_rejects_self_fork(self):
        response = self.client.post(
            "/history",
            json={"id": "self-fork", "name": "Self", "forkedFromConversationId": "self-fork"},
            headers=self.auth,
        )

        self.assertEqual(response.status_code, 400)

    def test_cross_user_access_is_forbidden(self):
        # Create as user A
        self._create_conv("private-c")
        # User B tries to read via different bearer
        response = self.client.get("/history/private-c", headers={"Authorization": "Bearer other-key"})
        self.assertEqual(response.status_code, 401)

    def test_legacy_replace_still_works(self):
        # The PUT path remains for the gateway's own chat-completion recording flow
        # which writes flat messages. Make sure it doesn't 500 after the schema migration.
        body = {"id": "legacy-c", "name": "Legacy", "lastModified": 1}
        self.client.post("/history", json=body, headers=self.auth)
        response = self.client.put(
            "/history/legacy-c",
            json={"title": "Updated", "messages": [{"role": "user", "content": "x"}]},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 200)
        conv = self.client.get("/history/legacy-c", headers=self.auth).json()
        self.assertEqual(conv["name"], "Updated")
        self.assertEqual(len(conv["messages"]), 1)

    def test_schema_migration_is_idempotent(self):
        # Second open of an existing DB must not error
        another = HistoryStore(self.db_path)
        # Sanity: the prior conversations are visible
        self._create_conv()
        rows = another.list_conversations_tree(entrypoint.identity_hash("auth.lost.plus:test-user"))
        self.assertEqual([r["id"] for r in rows], ["c1"])


if __name__ == "__main__":
    unittest.main()
