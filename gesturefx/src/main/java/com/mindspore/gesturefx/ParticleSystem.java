package com.mindspore.gesturefx;

import java.util.Random;

/**
 * SAT0RU-style particle physics engine.
 *
 * Trail: particles spawn AT fingertip position each frame (zero lag),
 *        old particles free-drift with velocity decay and fade.
 * Burst: direct velocity assignment, no force accumulation.
 *        Multi-layer: fast inner ring + slow outer ring.
 */
public class ParticleSystem {

    private static final int MAX_PARTICLES = 6000;

    // Per-particle flat arrays
    private final float[] px, py;
    private final float[] vx, vy;
    private final float[] life, maxLife;
    private final float[] cr, cg, cb;     // current color (evolves over time)
    private final float[] tr, tg, tb;     // target color (fades toward)
    private final float[] size;           // individual particle size multiplier
    private final int[] state;            // 0=dead, 1=trail, 2=explosion, 3=collapse, 4=beam
    private final int[] handIdx;
    private int activeCount;

    private final Random rnd = new Random();

    // ---- Physics ----
    private static final float TRAIL_DAMPING = 0.88f;
    private static final float BURST_DAMPING = 0.95f;
    private static final float TRAIL_LIFE = 1.2f;
    private static final float BURST_LIFE = 1.8f;
    private static final float BEAM_LIFE = 0.7f;

    // Burst speeds (pixels/sec)
    private static final float EXPLOSION_SPEED_FAST = 1800f;
    private static final float EXPLOSION_SPEED_SLOW = 600f;
    private static final float COLLAPSE_SPEED = 1200f;
    private static final float BEAM_SPEED = 2000f;

    // Spawn counts
    private static final int EXPLOSION_COUNT = 1000;
    private static final int COLLAPSE_COUNT = 800;
    private static final int BEAM_COUNT = 350;
    private static final int TRAIL_PER_FRAME_PER_POINT = 3;

    private float trailSpawnTimer;

    // ---- SAT0RU Color Palettes (vivid, multi-layered) ----

    // Trail: cyberpunk cyan→purple→pink
    private static final float[][] TRAIL_COLORS = {
        {0.0f, 1.0f, 1.0f},     // #00FFFF cyan
        {0.0f, 0.6f, 1.0f},     // #0099FF
        {0.3f, 0.0f, 1.0f},     // #4D00FF
        {0.7f, 0.0f, 1.0f},     // #B300FF violet
        {1.0f, 0.0f, 0.7f},     // #FF00B3 hot pink
        {0.0f, 1.0f, 0.4f},     // #00FF66 spring
    };

    // Explosion: 赫 Repulsion — white-hot flash → orange/gold/red
    private static final float[][] EXPLOSION_INNER = {
        {1.0f, 1.0f, 1.0f},     // white flash
        {1.0f, 0.95f, 0.9f},    // warm white
        {1.0f, 0.8f, 0.5f},     // peach
    };
    private static final float[][] EXPLOSION_OUTER = {
        {1.0f, 0.5f, 0.0f},     // #FF8000 orange
        {1.0f, 0.3f, 0.0f},     // #FF4D00 deep orange
        {1.0f, 0.15f, 0.0f},    // red-orange
        {1.0f, 0.7f, 0.1f},     // amber
    };

    // Collapse: 虚式茈 Purple Chaos — dark purple void + white core
    private static final float[][] COLLAPSE_CORE = {
        {1.0f, 1.0f, 1.0f},     // white
        {0.9f, 0.7f, 1.0f},     // lavender
        {0.7f, 0.5f, 1.0f},     // light purple
    };
    private static final float[][] COLLAPSE_OUTER = {
        {0.5f, 0.0f, 0.8f},     // #8000CC purple
        {0.3f, 0.0f, 0.6f},     // #4D0099 deep purple
        {0.15f, 0.0f, 0.4f},    // dark purple
        {0.05f, 0.0f, 0.2f},    // near-black purple
    };

    // Beam: bright cyan-white laser
    private static final float[][] BEAM_COLORS = {
        {1.0f, 1.0f, 1.0f},     // white core
        {0.5f, 1.0f, 1.0f},     // bright cyan
        {0.0f, 1.0f, 1.0f},     // cyan
        {0.0f, 0.7f, 1.0f},     // sky blue
    };

    // ---- Spawn point indices (full hand) ----
    private static final int[] TRAIL_SPAWN_INDICES = {
        4, 8, 12, 16, 20,       // fingertips
        5, 9, 13, 17,           // MCPs
        2, 6, 10, 14, 18,       // PIPs
        1,                      // thumb CMC
        0,                      // wrist
    };

    public ParticleSystem() {
        px = new float[MAX_PARTICLES];
        py = new float[MAX_PARTICLES];
        vx = new float[MAX_PARTICLES];
        vy = new float[MAX_PARTICLES];
        life = new float[MAX_PARTICLES];
        maxLife = new float[MAX_PARTICLES];
        cr = new float[MAX_PARTICLES];
        cg = new float[MAX_PARTICLES];
        cb = new float[MAX_PARTICLES];
        tr = new float[MAX_PARTICLES];
        tg = new float[MAX_PARTICLES];
        tb = new float[MAX_PARTICLES];
        size = new float[MAX_PARTICLES];
        state = new int[MAX_PARTICLES];
        handIdx = new int[MAX_PARTICLES];
    }

    // ===== Main Update =====

    public void update(float dt, float[][] targets, int[] gestures,
                       float[][] palmCenters, float[][] beamDirs) {

        // 1. Update existing particles (velocity + position + life)
        for (int i = 0; i < MAX_PARTICLES; i++) {
            if (life[i] <= 0) continue;
            int s = state[i];

            // Free drift with damping
            if (s == 1) {
                vx[i] *= TRAIL_DAMPING;
                vy[i] *= TRAIL_DAMPING;
            } else {
                vx[i] *= BURST_DAMPING;
                vy[i] *= BURST_DAMPING;
            }

            px[i] += vx[i] * dt;
            py[i] += vy[i] * dt;
            life[i] -= dt;

            // Color evolution: current → target
            float t = 1f - Math.min(1f, dt * 2f);
            cr[i] = cr[i] * t + tr[i] * (1f - t);
            cg[i] = cg[i] * t + tg[i] * (1f - t);
            cb[i] = cb[i] * t + tb[i] * (1f - t);
        }

        // 2. Spawn trail particles AT fingertip positions (zero lag)
        if (targets != null && gestures != null) {
            trailSpawnTimer += dt;
            float interval = 1f / 60f; // ~60fps spawn rate

            while (trailSpawnTimer >= interval) {
                trailSpawnTimer -= interval;
                for (int h = 0; h < targets.length; h++) {
                    if (targets[h] == null) continue;
                    int g = (h < gestures.length) ? gestures[h] : 0;
                    if (g != GestureDetector.GESTURE_DEFAULT) continue;

                    // Spawn at multiple keypoints
                    int kpCount = (targets[h].length - 2) / 2;
                    for (int s = 0; s < Math.min(5, kpCount); s++) {
                        int idx = s * 2 + 2;
                        if (idx + 1 < targets[h].length) {
                            for (int n = 0; n < TRAIL_PER_FRAME_PER_POINT; n++) {
                                spawnParticle(
                                    targets[h][idx] + jitter(),
                                    targets[h][idx + 1] + jitter(),
                                    0, 0, // no initial velocity
                                    TRAIL_LIFE, 1, h,
                                    randomFrom(TRAIL_COLORS),
                                    0.6f + rnd.nextFloat() * 0.8f);
                            }
                        }
                    }
                    // Also spawn at primary tip
                    for (int n = 0; n < TRAIL_PER_FRAME_PER_POINT * 2; n++) {
                        spawnParticle(
                            targets[h][0] + jitter(3f),
                            targets[h][1] + jitter(3f),
                            0, 0, TRAIL_LIFE, 1, h,
                            randomFrom(TRAIL_COLORS),
                            0.8f + rnd.nextFloat() * 1.2f);
                    }
                }
            }
        }

        // Count active
        activeCount = 0;
        for (int i = 0; i < MAX_PARTICLES; i++) {
            if (life[i] > 0) activeCount++;
        }
    }

    // ===== Gesture Trigger =====

    public void triggerEffect(int hIdx, int gesture, float cx, float cy,
                              float dirX, float dirY) {
        if (gesture == GestureDetector.GESTURE_OPEN_PALM) {
            triggerExplosion(hIdx, cx, cy);
        } else if (gesture == GestureDetector.GESTURE_FIST) {
            triggerCollapse(hIdx, cx, cy);
        } else if (gesture == GestureDetector.GESTURE_GUN) {
            triggerBeam(hIdx, cx, cy, dirX, dirY);
        }

        // Kill existing trail particles for this hand (replace with burst)
        for (int i = 0; i < MAX_PARTICLES; i++) {
            if (life[i] > 0 && handIdx[i] == hIdx && state[i] == 1) {
                life[i] = 0;
            }
        }
    }

    // ===== Burst Effects =====

    private void triggerExplosion(int hIdx, float cx, float cy) {
        // Inner ring: white-hot flash, fast
        for (int i = 0; i < EXPLOSION_COUNT / 3; i++) {
            float angle = rnd.nextFloat() * 6.2832f;
            float speed = EXPLOSION_SPEED_FAST * (0.7f + rnd.nextFloat() * 0.3f);
            spawnParticle(cx, cy,
                (float) Math.cos(angle) * speed,
                (float) Math.sin(angle) * speed,
                BURST_LIFE * 0.6f, 2, hIdx,
                randomFrom(EXPLOSION_INNER),
                2.0f + rnd.nextFloat() * 1.5f);
        }

        // Outer ring: orange/gold/red, slower
        for (int i = 0; i < EXPLOSION_COUNT * 2 / 3; i++) {
            float angle = rnd.nextFloat() * 6.2832f;
            float speed = EXPLOSION_SPEED_SLOW * (0.5f + rnd.nextFloat());
            spawnParticle(cx, cy,
                (float) Math.cos(angle) * speed,
                (float) Math.sin(angle) * speed,
                BURST_LIFE, 2, hIdx,
                randomFrom(EXPLOSION_OUTER),
                1.0f + rnd.nextFloat() * 2.0f);
        }
    }

    private void triggerCollapse(int hIdx, float cx, float cy) {
        // Core: white-purple, fast inward spiral
        for (int i = 0; i < COLLAPSE_COUNT / 3; i++) {
            float angle = rnd.nextFloat() * 6.2832f;
            float radius = 30f + rnd.nextFloat() * 80f;
            float sx = cx + (float) Math.cos(angle) * radius;
            float sy = cy + (float) Math.sin(angle) * radius;
            // Tangential velocity for orbit
            float tvx = -(float) Math.sin(angle) * COLLAPSE_SPEED * 0.3f;
            float tvy = (float) Math.cos(angle) * COLLAPSE_SPEED * 0.3f;
            // Inward velocity
            float ivx = -(float) Math.cos(angle) * COLLAPSE_SPEED * 0.7f;
            float ivy = -(float) Math.sin(angle) * COLLAPSE_SPEED * 0.7f;
            spawnParticle(sx, sy, tvx + ivx, tvy + ivy,
                BURST_LIFE * 0.8f, 3, hIdx,
                randomFrom(COLLAPSE_CORE),
                1.8f + rnd.nextFloat() * 1.5f);
        }

        // Outer ring: dark purple, wider spiral
        for (int i = 0; i < COLLAPSE_COUNT * 2 / 3; i++) {
            float angle = rnd.nextFloat() * 6.2832f;
            float radius = 80f + rnd.nextFloat() * 300f;
            float sx = cx + (float) Math.cos(angle) * radius;
            float sy = cy + (float) Math.sin(angle) * radius;
            float tvx = -(float) Math.sin(angle) * COLLAPSE_SPEED * 0.6f;
            float tvy = (float) Math.cos(angle) * COLLAPSE_SPEED * 0.6f;
            float ivx = -(float) Math.cos(angle) * COLLAPSE_SPEED * 0.4f;
            float ivy = -(float) Math.sin(angle) * COLLAPSE_SPEED * 0.4f;
            spawnParticle(sx, sy, tvx + ivx, tvy + ivy,
                BURST_LIFE, 3, hIdx,
                randomFrom(COLLAPSE_OUTER),
                0.8f + rnd.nextFloat() * 2.0f);
        }
    }

    private void triggerBeam(int hIdx, float cx, float cy, float dx, float dy) {
        for (int i = 0; i < BEAM_COUNT; i++) {
            // Tight cone around beam direction (±6°)
            float baseAngle = (float) Math.atan2(dy, dx);
            float spread = (rnd.nextFloat() - 0.5f) * 0.2f; // ~±6°
            float angle = baseAngle + spread;
            float speed = BEAM_SPEED * (0.5f + rnd.nextFloat() * 0.5f);
            spawnParticle(cx + jitter(4f), cy + jitter(4f),
                (float) Math.cos(angle) * speed,
                (float) Math.sin(angle) * speed,
                BEAM_LIFE, 4, hIdx,
                randomFrom(BEAM_COLORS),
                0.5f + rnd.nextFloat() * 1.2f);
        }
    }

    // ===== Vertex Packing =====

    /**
     * Pack active particles into float array for GPU.
     * Format: [x, y, z=0, w=1, r, g, b, a] per particle (8 floats).
     * Alpha = lifeRatio * size-based fading.
     */
    public int packVertexData(float[] buffer) {
        int count = 0;
        int max = Math.min(MAX_PARTICLES, buffer.length / 8);
        for (int i = 0; i < max; i++) {
            if (life[i] <= 0) continue;
            int off = count * 8;
            float lifeRatio = Math.max(0f, life[i] / Math.max(0.01f, maxLife[i]));
            buffer[off]     = px[i];
            buffer[off + 1] = py[i];
            buffer[off + 2] = 0f;
            buffer[off + 3] = 1f;
            buffer[off + 4] = cr[i];
            buffer[off + 5] = cg[i];
            buffer[off + 6] = cb[i];
            buffer[off + 7] = lifeRatio; // alpha from remaining life
            count++;
        }
        return count;
    }

    public int getActiveCount() { return activeCount; }

    // ===== Internal =====

    private void spawnParticle(float x, float y, float vx, float vy,
                               float lifeSec, int st, int hIdx,
                               float[] color, float sizeMul) {
        int idx = findSlot();
        if (idx < 0) return;
        px[idx] = x;
        py[idx] = y;
        this.vx[idx] = vx;
        this.vy[idx] = vy;
        life[idx] = lifeSec;
        maxLife[idx] = lifeSec;
        cr[idx] = color[0];
        cg[idx] = color[1];
        cb[idx] = color[2];
        tr[idx] = color[0];
        tg[idx] = color[1];
        tb[idx] = color[2];
        size[idx] = sizeMul;
        state[idx] = st;
        handIdx[idx] = hIdx;
    }

    private int findSlot() {
        // Find dead particle
        for (int i = 0; i < MAX_PARTICLES; i++) {
            if (life[i] <= 0) return i;
        }
        // Overwrite oldest (lowest life)
        int oldest = 0;
        float minL = life[0];
        for (int i = 1; i < MAX_PARTICLES; i++) {
            if (life[i] < minL) { minL = life[i]; oldest = i; }
        }
        return oldest;
    }

    private static float[] randomFrom(float[][] palette) {
        return palette[new Random().nextInt(palette.length)];
    }

    private float jitter() { return jitter(2f); }
    private float jitter(float scale) {
        return (rnd.nextFloat() - 0.5f) * scale;
    }
}
