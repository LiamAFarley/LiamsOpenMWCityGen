# 2026-08-24 Wall, Gate Road, and City Render Tools

## Updated tools

- `focus_wall_scene.py` can retain the complete scene while framing a selected
  wall arc, and can face the camera along a gatehouse's road-normal axis. This
  avoids diagnostic scenes that manufacture wall ends by deleting members.
- `render_wall_scatter_city.py` assembles the current city layout, seated stamp
  objects, composed wall, authored LAND/town ESP, and filtered regional scatter
  directly. `--view-direction X Y Z` and `--padding` produce comparable views
  at different angles and distances without editing placement artifacts.

Both tools are render-only. Inputs and camera values remain command-line data;
they do not change city, wall, terrain, or scatter generation.
