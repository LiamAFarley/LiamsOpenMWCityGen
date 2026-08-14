# Town-specific visual planner supplement

Fill this concise supplement before asking the generic planner to design a
specific town. It is an input brief, not a replacement for the visual canvas.

```yaml
town_id: "<stable id>"
design_reason: "<what this town must communicate>"
culture_and_kit_hierarchy:
  primary_kit: "<accepted kit id>"
  secondary_kits: ["<accepted kit id>"]
  civic_priority: ["<anchor type>"]
  commercial_priority: ["<anchor type>"]
required_anchors:
  - id: "<anchor id>"
    role: "<civic|commercial|gate|waterfront|craft>"
    door_intents: ["<public|service|private>"]
    visual_reason: "<why it belongs here>"
  - "<edge, agricultural, forest, water, or wilderness relationship>"
  - "<what must remain open or low density>"
local_design_intentions:
  - "<street/court/plaza intention>"
  - "<terrain/stair/burial intention>"
  - "<acceptable repeated motif and its written reason>"
iteration_limit: 3
```

The planner must still open the rendered image, preserve all source and
measured door evidence, keep hard errors separate from advisories, and explain
any override in the visual-plan extension.
