// Fragment shader for point sprite particles with radial glow
// Uses additive blending (set from Java side: GL_SRC_ALPHA, GL_ONE)
// gl_PointCoord ranges from (0,0) bottom-left to (1,1) top-right

precision mediump float;

uniform sampler2D uTexture;
varying vec4 vColor;

void main() {
    // Calculate distance from sprite center (0.5, 0.5)
    float dist = length(gl_PointCoord - vec2(0.5));

    // Radial falloff: bright center, fading to transparent edge
    float glow = 1.0 - smoothstep(0.0, 0.5, dist);

    // Sample glow texture for soft edge
    vec4 texColor = texture2D(uTexture, gl_PointCoord);

    // Combine texture, color, and radial glow
    float alpha = glow * texColor.a * vColor.a;
    vec3 rgb = vColor.rgb * texColor.rgb;

    gl_FragColor = vec4(rgb, alpha);
}
