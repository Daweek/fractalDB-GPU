#version 330 core

in vec2 UV;

out vec3 color;

uniform sampler2DRect renderedTexture;


void main(){
	color = texture(renderedTexture,gl_FragCoord.xy).xyz;
}
