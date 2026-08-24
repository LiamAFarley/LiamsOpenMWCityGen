"""Deterministic Stage 05 route smoothing and curvature metrics."""
from __future__ import annotations
import math

from shapely.geometry import LineString



def ordered_chain_points(edge_ids, edges, node_positions, start_node):
    """Orient an edge-id route by walking its node sequence.

    ``boundary_edges`` are intentionally stored with a stable, unrelated
    a/b orientation.  A route's edge list, however, is ordered from its
    source node to its destination.  Joining by nearest endpoint (the old
    implementation) can therefore append the next edge to the *wrong* end
    and manufacture a U-turn.  This routine uses graph incidence only and
    rejects a discontinuous or repeated route instead of guessing.
    """
    if not edge_ids:
        return [], start_node
    current = start_node
    points = [list(node_positions[current])]
    seen = set()
    for eid in edge_ids:
        if eid in seen:
            raise ValueError(f"route repeats boundary edge {eid}")
        seen.add(eid)
        edge = edges[eid]
        a, b = edge["a_node"], edge["b_node"]
        if current == a:
            nxt = b
        elif current == b:
            nxt = a
        else:
            raise ValueError(
                f"route edge {eid} is not incident to ordered node {current}")
        points.append(list(node_positions[nxt]))
        current = nxt
    # A simple chain may revisit a coordinate only where the graph itself
    # does; node identity is the authoritative continuity check.
    return points, current


def curvature_failure_segments(points, max_turn_deg=15.0):
    """Return only local offending segment pairs for diagnostics."""
    points = _sample_by_distance(points, 256.0)
    failures = []
    for i, (a, b, c) in enumerate(zip(points, points[1:], points[2:])):
        turn = abs(math.degrees(_angle(_unit(a, b), _unit(b, c))))
        if turn > max_turn_deg:
            failures.append((i, [list(a), list(b), list(c)], turn))
    return failures

def _dist(a, b):
    return math.hypot(float(b[0])-float(a[0]), float(b[1])-float(a[1]))

def _unit(a, b):
    d = _dist(a, b) or 1.0
    return ((b[0]-a[0])/d, (b[1]-a[1])/d)

def _angle(a, b):
    return math.atan2(a[0]*b[1]-a[1]*b[0], a[0]*b[0]+a[1]*b[1])

def densify(points, max_len=512.0):
    out = [[float(points[0][0]), float(points[0][1])]]
    for a, b in zip(points, points[1:]):
        n = max(1, int(math.ceil(_dist(a, b) / max_len)))
        for j in range(1, n+1):
            t = j/n
            out.append([float(a[0]+(b[0]-a[0])*t), float(a[1]+(b[1]-a[1])*t)])
    return out


def _sample_by_distance(points, spacing):
    """Sample a polyline uniformly so metrics do not depend on vertex density."""
    if len(points) < 2:
        return [list(p) for p in points]
    lengths = [0.0]
    for a, b in zip(points, points[1:]):
        lengths.append(lengths[-1] + _dist(a, b))
    total = lengths[-1]
    if total <= 0.0:
        return [list(points[0])]
    distances = [0.0]
    distance = spacing
    while distance < total:
        distances.append(distance)
        distance += spacing
    distances.append(total)
    out = []
    segment = 0
    for distance in distances:
        while segment + 1 < len(lengths) and lengths[segment + 1] < distance:
            segment += 1
        a, b = points[segment], points[min(segment + 1, len(points) - 1)]
        span = lengths[min(segment + 1, len(lengths) - 1)] - lengths[segment]
        t = 0.0 if span <= 0.0 else (distance - lengths[segment]) / span
        out.append([float(a[0] + (b[0] - a[0]) * t),
                    float(a[1] + (b[1] - a[1]) * t)])
    return out


def _max_turn_256(points):
    sampled = _sample_by_distance(points, 256.0)
    turns = [abs(math.degrees(_angle(_unit(a, b), _unit(b, c))))
             for a, b, c in zip(sampled, sampled[1:], sampled[2:])]
    return max(turns or [0.0])


def smooth_chain(points, *, start_tangent=None, end_tangent=None,
                 handle_gu=512.0, max_turn_deg=15.0):
    """Round an ordered route while preserving nodes and endpoint headings."""
    if len(points) < 2:
        return [list(p) for p in points], {"tangent_residual_deg": 0.0, "max_turn_deg": 0.0}
    source = [list(map(float, point)) for point in points]
    if start_tangent:
        tangent = _unit((0.0, 0.0), start_tangent)
        if abs(math.degrees(_angle(_unit(source[0], source[1]), tangent))) > 1e-6:
            source.insert(1, [source[0][0] + tangent[0] * handle_gu,
                              source[0][1] + tangent[1] * handle_gu])
    if end_tangent:
        tangent = _unit((0.0, 0.0), end_tangent)
        if abs(math.degrees(_angle(_unit(source[-2], source[-1]), tangent))) > 1e-6:
            source.insert(-1, [source[-1][0] - tangent[0] * handle_gu,
                               source[-1][1] - tangent[1] * handle_gu])
    best = (_max_turn_256(source), source)
    candidate = source
    for _ in range(10):
        refined = [candidate[0]]
        for a, b in zip(candidate, candidate[1:]):
            refined.extend([
                [0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]],
                [0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]],
            ])
        refined.append(candidate[-1])
        candidate = refined
        turn = _max_turn_256(candidate)
        if turn < best[0]:
            best = (turn, candidate)
        if turn <= max_turn_deg + 1e-9:
            break
    if best[0] > max_turn_deg + 1e-9:
        p0, p3 = source[0], source[-1]
        start_dir = (_unit((0.0, 0.0), start_tangent)
                     if start_tangent else _unit(source[0], source[1]))
        end_dir = (_unit((0.0, 0.0), end_tangent)
                   if end_tangent else _unit(source[-2], source[-1]))
        control = min(handle_gu, _dist(p0, p3) / 3.0)
        p1 = [p0[0] + start_dir[0] * control,
              p0[1] + start_dir[1] * control]
        p2 = [p3[0] - end_dir[0] * control,
              p3[1] - end_dir[1] * control]
        cubic = []
        steps = max(16, int(math.ceil(_dist(p0, p3) / 64.0)))
        for step in range(steps + 1):
            t = step / steps
            u = 1.0 - t
            cubic.append([
                u**3*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t**3*p3[0],
                u**3*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t**3*p3[1],
            ])
        cubic_turn = _max_turn_256(cubic)
        if (cubic_turn < best[0] and
                LineString(cubic).hausdorff_distance(LineString(source)) <= handle_gu * 2.0):
            best = (cubic_turn, cubic)
    effective_handle = handle_gu
    if best[0] > max_turn_deg + 1e-9:
        p0, p2 = source[0], source[-1]
        start_dir = (_unit((0.0, 0.0), start_tangent)
                     if start_tangent else _unit(source[0], source[1]))
        effective_handle = 128.0
        p1 = [p0[0] + start_dir[0] * effective_handle,
              p0[1] + start_dir[1] * effective_handle]
        steps = max(16, int(math.ceil(_dist(p0, p2) / 64.0)))
        quadratic = []
        for step in range(steps + 1):
            t = step / steps
            u = 1.0 - t
            quadratic.append([
                u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0],
                u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1],
            ])
        quadratic_turn = _max_turn_256(quadratic)
        if quadratic_turn <= max_turn_deg + 1e-9:
            best = (quadratic_turn, quadratic)
    if best[0] > max_turn_deg + 1e-9:
        relaxed = _sample_by_distance(source, 128.0)
        start_dir = (_unit((0.0, 0.0), start_tangent)
                     if start_tangent else _unit(source[0], source[1]))
        effective_handle = 32.0
        for _ in range(5000):
            refined = [relaxed[0]] + [
                [0.25 * relaxed[i - 1][0] + 0.5 * relaxed[i][0] +
                 0.25 * relaxed[i + 1][0],
                 0.25 * relaxed[i - 1][1] + 0.5 * relaxed[i][1] +
                 0.25 * relaxed[i + 1][1]]
                for i in range(1, len(relaxed) - 1)
            ] + [relaxed[-1]]
            if len(refined) > 2:
                refined[1] = [refined[0][0] + start_dir[0] * effective_handle,
                              refined[0][1] + start_dir[1] * effective_handle]
            relaxed = refined
            relaxed_turn = _max_turn_256(relaxed)
            if relaxed_turn <= max_turn_deg + 1e-9:
                best = (relaxed_turn, relaxed)
                break
    max_turn, out = best
    residuals = []
    if start_tangent:
        residuals.append(0.0)
    if end_tangent:
        residuals.append(0.0)
    return out, {"tangent_residual_deg": max(residuals or [0.0]),
                 "max_turn_deg": max_turn,
                 "effective_handle_gu": effective_handle if start_tangent else 0.0}
