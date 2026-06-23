package com.mindspore.gesturefx;

import java.util.Random;

/**
 * Particle system handling physics for 3000+ particles.
 * Manages particle lifecycle, spring-damping physics,
 * and gesture-driven behaviors (trail, explosion, collapse, beam).
 */
public class ParticleSystem {

    // ---- Particle structure (flat arrays for GPU upload efficiency) ----
    private static final int MAX_PARTICLES = 5000;

    // Per-particle data
    private final float[] posX, posY;
    private final float[] velX, velY;
    private final float[] life;       // remaining life in seconds
    private final float[] maxLife;
    private final float[] colorR, colorG, colorB;
    private final int[] state;        // 0=dead, 1=trail, 2=explosion, 3=collapse, 4=beam
    private final int[] handIndex;    // which hand owns this particle

    private int activeCount = 0;
    private final Random random = new Random();

    // ---- Physics constants (increased for more dynamic effects) ----
    private static final float SPRING_K = 35.0f;       // spring stiffness for trail (was 28)
    private static final float DAMPING = 0.82f;         // velocity damping per frame (was 0.85)
    private static final float SPAWN_RATE = 120f;       // particles per second per spawn point (was 80)
    private static final float MAX_LIFE = 2.0f;         // seconds
    private static final float MIN_LIFE = 0.5f;
    private static final float EXPLOSION_FORCE = 1400f;  // explosion radial force (was 900)
    private static final float COLLAPSE_FORCE = 1000f;   // collapse inward force (was 700)
    private static final float BEAM_SPEED = 1600f;       // beam particle speed (was 1200)
    private static final float BEAM_ANGLE = 8f;          // beam cone half-angle (tighter)
    private static final float VORTEX_FORCE = 600f;      // vortex rotational force

    // ---- Keypoint spawn indices for full hand coverage ----
    // Fingertips
    private static final int[] TIP_INDICES = {4, 8, 12, 16, 20};
    // MCP joints (knuckles)
    private static final int[] MCP_INDICES = {1, 5, 9, 13, 17};
    // PIP joints (middle knuckles)
    private static final int[] PIP_INDICES = {2, 6, 10, 14, 18};

    // ---- Accumulator for frame-rate independent spawning ----
    private float spawnAccum = 0f;

    // ---- Color palettes (SAT0RU-inspired, vivid) ----
    // Trail: cyberpunk cyan-purple-pink
    private static final float[][] TRAIL_COLORS = {
        {0.0f, 1.0f, 1.0f},     // cyan #00FFFF
        {0.0f, 0.7f, 1.0f},     // light blue #00B3FF
        {0.4f, 0.0f, 1.0f},     // purple #6600FF
        {0.7f, 0.0f, 1.0f},     // violet #B300FF
        {1.0f, 0.0f, 0.8f},     // hot pink #FF00CC
        {0.0f, 1.0f, 0.5f},     // spring green #00FF80
    };
    // Explosion: fire orange-gold-white
    private static final float[][] EXPLOSION_COLORS = {
        {1.0f, 0.3f, 0.0f},     // deep orange #FF4D00
        {1.0f, 0.5f, 0.0f},     // orange #FF8000
        {1.0f, 0.7f, 0.0f},     // amber #FFB300
        {1.0f, 0.9f, 0.1f},     // gold #FFE61A
        {1.0f, 1.0f, 0.4f},     // light yellow #FFFF66
        {1.0f, 0.6f, 1.0f},     // warm pink
    };
    // Collapse: blue-violet-white core
    private static final float[][] COLLAPSE_COLORS = {
        {1.0f, 1.0f, 1.0f},     // white core
        {0.3f, 0.6f, 1.0f},     // light blue #4D99FF
        {0.0f, 0.3f, 1.0f},     // blue #004DFF
        {0.3f, 0.0f, 1.0f},     // indigo #4D00FF
        {0.6f, 0.0f, 1.0f},     // violet #9900FF
        {0.0f, 0.8f, 1.0f},     // sky blue
    };
    // Beam: bright cyan-white laser
    private static final float[][] BEAM_COLORS = {
        {0.0f, 1.0f, 1.0f},     // cyan #00FFFF
        {0.2f, 0.9f, 1.0f},     // pale cyan
        {0.5f, 1.0f, 1.0f},     // aqua
        {0.8f, 1.0f, 1.0f},     // near white
        {1.0f, 1.0f, 1.0f},     // white core
        {0.0f, 0.8f, 1.0f},     // cerulean
    };

    public ParticleSystem() {
        posX = new float[MAX_PARTICLES];
        posY = new float[MAX_PARTICLES];
        velX = new float[MAX_PARTICLES];
        velY = new float[MAX_PARTICLES];
        life = new float[MAX_PARTICLES];
        maxLife = new float[MAX_PARTICLES];
        colorR = new float[MAX_PARTICLES];
        colorG = new float[MAX_PARTICLES];
        colorB = new float[MAX_PARTICLES];
        state = new int[MAX_PARTICLES];
        handIndex = new int[MAX_PARTICLES];
    }

    /**
     * Update all active particles.
     * @param deltaSeconds time since last frame
     * @param targets array of [handIndex][targetX, targetY] for trail attractors
     * @param gestureTypes array of [handIndex] gesture type for each hand
     * @param palmCenters array of [handIndex][centerX, centerY] for explosion/collapse center
     * @param beamDirections array of [handIndex][dirX, dirY] for beam direction
     */
    public void update(float deltaSeconds,
                       float[][] targets,
                       int[] gestureTypes,
                       float[][] palmCenters,
                       float[][] beamDirections) {

        // Update existing particles
        for (int i = 0; i < MAX_PARTICLES; i++) {
            if (life[i] <= 0) continue;

            int s = state[i];
            int hIdx = handIndex[i];

            if (s == 1) {
                // Trail: spring-damping toward target fingertip
                if (targets != null && hIdx < targets.length && targets[hIdx] != null) {
                    float tx = targets[hIdx][0];
                    float ty = targets[hIdx][1];
                    float dx = tx - posX[i];
                    float dy = ty - posY[i];
                    velX[i] += dx * SPRING_K * deltaSeconds;
                    velY[i] += dy * SPRING_K * deltaSeconds;
                }
            } else if (s == 2) {
                // Explosion (赫 Repulsion): radial outward + vortex swirl
                if (palmCenters != null && hIdx < palmCenters.length && palmCenters[hIdx] != null) {
                    float dx = posX[i] - palmCenters[hIdx][0];
                    float dy = posY[i] - palmCenters[hIdx][1];
                    float dist = (float) Math.sqrt(dx * dx + dy * dy);
                    if (dist > 1f) {
                        // Radial outward force
                        velX[i] += (dx / dist) * EXPLOSION_FORCE * deltaSeconds;
                        velY[i] += (dy / dist) * EXPLOSION_FORCE * deltaSeconds;
                        // Tangential vortex force (perpendicular to radial)
                        velX[i] += (-dy / dist) * VORTEX_FORCE * deltaSeconds;
                        velY[i] += (dx / dist) * VORTEX_FORCE * deltaSeconds;
                    }
                }
            } else if (s == 3) {
                // Collapse (虚式茈 Purple Chaos): spiral inward + vortex
                if (palmCenters != null && hIdx < palmCenters.length && palmCenters[hIdx] != null) {
                    float dx = palmCenters[hIdx][0] - posX[i];
                    float dy = palmCenters[hIdx][1] - posY[i];
                    float dist = (float) Math.sqrt(dx * dx + dy * dy);
                    if (dist > 1f) {
                        // Radial inward force
                        velX[i] += (dx / dist) * COLLAPSE_FORCE * deltaSeconds;
                        velY[i] += (dy / dist) * COLLAPSE_FORCE * deltaSeconds;
                        // Tangential vortex force (creates swirl)
                        velX[i] += (-dy / dist) * VORTEX_FORCE * 0.7f * deltaSeconds;
                        velY[i] += (dx / dist) * VORTEX_FORCE * 0.7f * deltaSeconds;
                    } else {
                        // Near center: tight orbit
                        velX[i] += -dy * 5f * deltaSeconds;
                        velY[i] += dx * 5f * deltaSeconds;
                    }
                }
            } else if (s == 4) {
                // Collapse: radial inward force toward palm center
                if (palmCenters != null && hIdx < palmCenters.length && palmCenters[hIdx] != null) {
                    float dx = palmCenters[hIdx][0] - posX[i];
                    float dy = palmCenters[hIdx][1] - posY[i];
                    float dist = (float) Math.sqrt(dx * dx + dy * dy);
                    if (dist > 1f) {
                        velX[i] += (dx / dist) * COLLAPSE_FORCE * deltaSeconds;
                        velY[i] += (dy / dist) * COLLAPSE_FORCE * deltaSeconds;
                    }
                }
            } else if (s == 4) {
                // Beam: constant velocity in beam direction
                // Velocity is set at spawn time, just let it fly
            }

            // Apply velocity
            velX[i] *= DAMPING;
            velY[i] *= DAMPING;
            posX[i] += velX[i] * deltaSeconds;
            posY[i] += velY[i] * deltaSeconds;

            // Decay life
            life[i] -= deltaSeconds;
        }

        // Spawn new trail particles at targets
        if (targets != null && gestureTypes != null) {
            spawnAccum += deltaSeconds;
            float spawnInterval = 1f / SPAWN_RATE;

            while (spawnAccum >= spawnInterval) {
                spawnAccum -= spawnInterval;
                for (int hIdx = 0; hIdx < targets.length; hIdx++) {
                    if (targets[hIdx] != null) {
                        int gType = (gestureTypes != null && hIdx < gestureTypes.length)
                            ? gestureTypes[hIdx] : GestureDetector.GESTURE_DEFAULT;

                        // Only spawn trails in default mode
                        if (gType == GestureDetector.GESTURE_DEFAULT) {
                            // Spawn from multiple keypoints for full hand coverage
                            if (targets[hIdx].length > 2) {
                                // targets[hIdx] contains [tipX, tipY, ... keypoint Xs and Ys]
                                // Spawn at a random subset of provided positions
                                int kpCount = (targets[hIdx].length - 2) / 2;
                                for (int k = 0; k < Math.min(3, kpCount); k++) {
                                    int idx = k * 2 + 2;
                                    if (idx + 1 < targets[hIdx].length) {
                                        spawnParticle(
                                            targets[hIdx][idx] + randomOffset() * 3f,
                                            targets[hIdx][idx + 1] + randomOffset() * 3f,
                                            1, hIdx);
                                    }
                                }
                            }
                            // Also spawn at the primary tip position
                            spawnParticle(targets[hIdx][0] + randomOffset() * 5f,
                                         targets[hIdx][1] + randomOffset() * 5f,
                                         1, hIdx);
                        }
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

    /**
     * Apply a new gesture mode to particles.
     * Transitions existing trail particles to the new gesture effect,
     * or spawns new particles if needed.
     */
    public void applyGesture(int hIdxParam, int newGesture,
                             float palmCenterX, float palmCenterY,
                             float beamDirX, float beamDirY) {

        if (newGesture == GestureDetector.GESTURE_DEFAULT) {
            // Nothing to change, particles stay in trail mode
            return;
        }

        int targetState;
        float[][] colorPalette;

        switch (newGesture) {
            case GestureDetector.GESTURE_OPEN_PALM:
                targetState = 2;
                colorPalette = EXPLOSION_COLORS;
                break;
            case GestureDetector.GESTURE_FIST:
                targetState = 3;
                colorPalette = COLLAPSE_COLORS;
                break;
            case GestureDetector.GESTURE_GUN:
                targetState = 4;
                colorPalette = BEAM_COLORS;
                break;
            default:
                return;
        }

        // Transition existing particles of this hand to new state
        for (int i = 0; i < MAX_PARTICLES; i++) {
            if (life[i] > 0 && handIndex[i] == hIdxParam) {
                state[i] = targetState;
                // Randomize color for visual variety
                float[] col = colorPalette[random.nextInt(colorPalette.length)];
                colorR[i] = col[0];
                colorG[i] = col[1];
                colorB[i] = col[2];
                // Reset life for fresh effect
                life[i] = MAX_LIFE * (0.5f + random.nextFloat() * 0.5f);
                maxLife[i] = life[i];

                if (targetState == 2) {
                    // Explosion: set velocity outward from palm
                    float dx = posX[i] - palmCenterX;
                    float dy = posY[i] - palmCenterY;
                    float dist = (float) Math.sqrt(dx * dx + dy * dy);
                    if (dist > 0.1f) {
                        velX[i] = (dx / dist) * EXPLOSION_FORCE * 0.5f;
                        velY[i] = (dy / dist) * EXPLOSION_FORCE * 0.5f;
                    } else {
                        float angle = random.nextFloat() * 6.2832f;
                        velX[i] = (float) Math.cos(angle) * EXPLOSION_FORCE * 0.5f;
                        velY[i] = (float) Math.sin(angle) * EXPLOSION_FORCE * 0.5f;
                    }
                } else if (targetState == 3) {
                    // Collapse: set velocity inward to palm
                    float dx = palmCenterX - posX[i];
                    float dy = palmCenterY - posY[i];
                    velX[i] = dx * 3f;
                    velY[i] = dy * 3f;
                } else if (targetState == 4) {
                    // Beam: set direction from index tip
                    float angle = (float) Math.atan2(beamDirY, beamDirX);
                    angle += (random.nextFloat() - 0.5f) * (BEAM_ANGLE * 0.01745f * 2f);
                    velX[i] = (float) Math.cos(angle) * BEAM_SPEED;
                    velY[i] = (float) Math.sin(angle) * BEAM_SPEED;
                }
            }
        }

        // Spawn extra particles at palm center for explosion/collapse
        if (targetState == 2 || targetState == 3) {
            int extraCount = 600;  // SAT0RU-level burst (was 200)
            float[][] palette = (targetState == 2) ? EXPLOSION_COLORS : COLLAPSE_COLORS;
            for (int i = 0; i < extraCount; i++) {
                float ox = palmCenterX + randomOffset() * 30f;
                float oy = palmCenterY + randomOffset() * 30f;
                float[] col = palette[random.nextInt(palette.length)];
                int idx = findDeadParticle();
                if (idx >= 0) {
                    posX[idx] = ox;
                    posY[idx] = oy;
                    if (targetState == 2) {
                        // Explosion burst: random radial outward
                        float angle = random.nextFloat() * 6.2832f;
                        float speed = EXPLOSION_FORCE * (0.3f + random.nextFloat() * 0.7f);
                        velX[idx] = (float) Math.cos(angle) * speed;
                        velY[idx] = (float) Math.sin(angle) * speed;
                    } else {
                        // Collapse: spiral inward from random starting orbit
                        float angle = random.nextFloat() * 6.2832f;
                        float radius = 50f + random.nextFloat() * 200f;
                        posX[idx] = palmCenterX + (float) Math.cos(angle) * radius;
                        posY[idx] = palmCenterY + (float) Math.sin(angle) * radius;
                        // Tangential velocity for spiral + slight inward
                        velX[idx] = -(posY[idx] - palmCenterY) * 3f;
                        velY[idx] = (posX[idx] - palmCenterX) * 3f;
                    }
                    life[idx] = 0.5f + random.nextFloat() * 1.5f;
                    maxLife[idx] = life[idx];
                    colorR[idx] = col[0];
                    colorG[idx] = col[1];
                    colorB[idx] = col[2];
                    state[idx] = targetState;
                    handIndex[idx] = hIdxParam;
                }
            }
        }

        // Spawn beam particles at index tip
        if (targetState == 4) {
            for (int i = 0; i < 250; i++) {  // was 100
                int idx = findDeadParticle();
                if (idx >= 0) {
                    posX[idx] = palmCenterX; // palmCenterX is actually index tip for beam
                    posY[idx] = palmCenterY;
                    float angle = (float) Math.atan2(beamDirY, beamDirX);
                    angle += (random.nextFloat() - 0.5f) * (BEAM_ANGLE * 0.01745f * 2f);
                    float speed = BEAM_SPEED * (0.5f + random.nextFloat());
                    velX[idx] = (float) Math.cos(angle) * speed;
                    velY[idx] = (float) Math.sin(angle) * speed;
                    float[] col = BEAM_COLORS[random.nextInt(BEAM_COLORS.length)];
                    colorR[idx] = col[0];
                    colorG[idx] = col[1];
                    colorB[idx] = col[2];
                    life[idx] = 0.3f + random.nextFloat() * 0.7f;
                    maxLife[idx] = life[idx];
                    state[idx] = 4;
                    handIndex[idx] = hIdxParam;
                }
            }
        }
    }

    /**
     * Pack active particles into a float array for GPU upload.
     * Format: [x, y, z=0, w=1, r, g, b, a] per particle (8 floats).
     * Alpha is computed from remaining life ratio.
     * @return number of active particles packed
     */
    public int packVertexData(float[] buffer) {
        int count = 0;
        for (int i = 0; i < MAX_PARTICLES && count * 8 < buffer.length; i++) {
            if (life[i] > 0) {
                int offset = count * 8;
                buffer[offset]     = posX[i];
                buffer[offset + 1] = posY[i];
                buffer[offset + 2] = 0f;    // z
                buffer[offset + 3] = 1f;    // w
                buffer[offset + 4] = colorR[i];
                buffer[offset + 5] = colorG[i];
                buffer[offset + 6] = colorB[i];
                // Alpha fades as life decreases
                float lifeRatio = Math.max(0f, life[i] / Math.max(0.01f, maxLife[i]));
                buffer[offset + 7] = lifeRatio;
                count++;
            }
        }
        return count;
    }

    public int getActiveCount() {
        return activeCount;
    }

    // ---- Internal helpers ----

    private void spawnParticle(float x, float y, int st, int hIdx) {
        int idx = findDeadParticle();
        if (idx < 0) return;

        posX[idx] = x;
        posY[idx] = y;
        velX[idx] = randomOffset() * 50f;
        velY[idx] = randomOffset() * 50f;
        life[idx] = MIN_LIFE + random.nextFloat() * (MAX_LIFE - MIN_LIFE);
        maxLife[idx] = life[idx];
        state[idx] = st;
        handIndex[idx] = hIdx;

        // Random trail color
        float[] col = TRAIL_COLORS[random.nextInt(TRAIL_COLORS.length)];
        colorR[idx] = col[0];
        colorG[idx] = col[1];
        colorB[idx] = col[2];
    }

    private int findDeadParticle() {
        // Try finding a dead particle
        for (int i = 0; i < MAX_PARTICLES; i++) {
            if (life[i] <= 0) return i;
        }
        // All alive, replace oldest (lowest life)
        int oldest = 0;
        float minLife = life[0];
        for (int i = 1; i < MAX_PARTICLES; i++) {
            if (life[i] < minLife) {
                minLife = life[i];
                oldest = i;
            }
        }
        return oldest;
    }

    private float randomOffset() {
        return (random.nextFloat() - 0.5f) * 2f;
    }
}
