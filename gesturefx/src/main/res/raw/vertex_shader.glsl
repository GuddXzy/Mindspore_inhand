// Vertex shader for point sprite particles
// Transforms particle positions from world space to clip space
// Uses orthographic projection for pixel-perfect mapping

uniform mat4 uMVPMatrix;
uniform float uPointSize;

attribute vec4 aPosition;
attribute vec4 aColor;

varying vec4 vColor;

void main() {
    gl_Position = uMVPMatrix * aPosition;
    gl_PointSize = uPointSize;
    vColor = aColor;
}
