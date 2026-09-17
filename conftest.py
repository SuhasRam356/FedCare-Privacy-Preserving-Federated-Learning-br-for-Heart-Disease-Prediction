# conftest.py – Prevent pytest from collecting fedcare/task.py::test()
# as a test function (it's the evaluation helper, not a test).
collect_ignore_glob = ["fedcare/*"]
