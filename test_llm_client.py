"""Regression test for tokenizer initialization; uses only the standard library."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import llm_client as m


class TokenizerInitializationTest(unittest.TestCase):
    def test_truncate_initialization_respects_lock(self):
        observed = threading.Event()
        gate = threading.Lock()
        loads = []

        class ObservedLock:
            def __enter__(self):
                observed.set()
                gate.acquire()

            def __exit__(self, *args):
                gate.release()

        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": list(text)}

            def decode(self, ids):
                return "".join(ids)

        def load():
            loads.append(1)
            observed.set()
            return Tokenizer()

        with patch.multiple(
            m,
            _tokenizer=None,
            _tokenizer_loaded=False,
            _tokenizer_lock=ObservedLock(),
            _load_tokenizer=load,
        ):
            gate.acquire()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(m.truncate_to_tokens, "abc", 2)
                try:
                    self.assertTrue(observed.wait(3), "Worker did not reach initialization")
                    self.assertEqual(loads, [], "Tokenizer loaded while initialization lock was held")
                finally:
                    gate.release()
                self.assertEqual(future.result(timeout=3), "ab")
            self.assertEqual(loads, [1])


if __name__ == "__main__":
    unittest.main()
