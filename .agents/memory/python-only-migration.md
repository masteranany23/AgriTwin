---
name: Python-only migration
description: How to handle an imported Python backend when the workspace migration scaffold targets JavaScript services.
---

When an imported repository is Python-only, the JavaScript migration detector may report no routes, schema, or frontend even though the project contains a substantial backend. Preserve the Python source, tests, migrations, and documentation as a separate workspace copy rather than translating scientific code into the generated JavaScript API scaffold.

**Why:** The migration scaffold is optimized for JS/TS full-stack projects, while scientific Python behavior is explicitly frozen and must not be rewritten just to fit the scaffold.

**How to apply:** Keep the dedicated web artifact separate for future UI work, document the preserved Python run command, and verify copied source trees against the migration backup.