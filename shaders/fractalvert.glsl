#version 430 core

layout (location = 0) in vec2 position;
layout (location = 1) in vec2 color;


out vec4 vColor;


uniform int numMappings;
uniform mat4 projection;
uniform vec2 rotation;
uniform int rtype;
uniform vec2 rnd;
 
//uniform vec2 translation;

float random (vec2 st) {
    return fract(sin(dot(st.xy,vec2(10,78.233)))*43758.5453123);
    //return fract(dot(st.xy,vec2(1.60,0.2))*3.0);
}

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main()
{
    
    //vec2 transformedPosition = (scaling * position + translation)*vec2(1.0,1.0);
    //vec2 transformedPosition = position * vec2(rotation.x,rotation.y);
    
    vec4 original = projection * vec4(position, 0.0, 1.0);
    gl_Position = original * vec4(rotation,0.0,1.0);

    vec3 rgbColor = vec3(0.9,0.9,0.0);

    // rType :  0,1 Point Gray, Patch Gray...
    //          2,3 Point Color, Patch Color...
    
    if(rtype == 2 || rtype == 3){
        vec3 hsvColor = vec3(color.y/numMappings, 1.0, 1.0);
        rgbColor = hsv2rgb(hsvColor);
    }
    else{
        rgbColor = vec3(0.5,0.5,0.5);
    }
   
    
    
    // Gray color
    
    
    vColor = vec4(rgbColor, 1.0);
}
