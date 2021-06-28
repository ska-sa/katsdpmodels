Revision history
================

0.3
---
- Add support for primary beam models (#26, #28-#34)
- Update to newer `aiohttp_retry`_ API (#27)

.. _aiohttp_retry: https://github.com/inyutin/aiohttp_retry

0.2
---
- Breaking change: rename the `channel_bandwidth` parameter to
  :meth:`.RFIMask.is_masked` and :meth:`.RFIMask.max_baseline_length` for
  consistency with other classes.
- Fix type annotation issues when used with numpy 1.20.
- Improve the user documentation.

0.1
---
- First public release
