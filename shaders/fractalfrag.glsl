#version 430 core

in vec4 vColor;
out vec4 fColor;

//in int rtype;
uniform int rtype;
uniform vec2 rnd;

float rand(vec2 co)
{
    float a = 12.9898*(rnd.x*10);
    float b = 78.233*(rnd.y*10);
    float c = 43758.5453;
    float dt= dot(co.xy ,vec2(a,b));
    float sn= mod(dt,3.14);
    return fract(sin(sn) * c);
}

void main()
{
    
    vec4 gray = vec4(0.5,0.5,0.5,1.0);
    //vec4 gray = vec4(rnd.x,rnd.y,rnd.x*rnd.y,1.0);
    vec4 black = vec4(0.0,0.0,0.0,1.0);

    if (rtype == 1){
        if(rand(gl_FragCoord.xy)<0.5) 
            fColor = gray;
        else 
            fColor = black;
    } 
    else{
       fColor = vColor;
    } 
      
    //fColor = vColor;           
}

/*
float rn(float xx){         
    float x0=floor(xx);
    float x1=x0+1;
    float v0 = fract(sin (x0*.014686)*31718.927+x0);
    float v1 = fract(sin (x1*.014686)*31718.927+x1);          

    return (v0*(1-fract(xx))+v1*(fract(xx)))*2-1*sin(xx);
}

float random (vec2 st) {
    //return fract(sin(dot(st.xy,vec2(10,78.233)))*43758.5453123);
    
    //return fract(sin(dot(st.xy,vec2(1.6,0.233)))*300.0);
    //return fract(dot(st.xy,vec2(0.2,0.233)));
    //return fract(dot(st.xy,vec2(0.5,0.5)));
    return fract(dot(st.xy,rnd));
    //return dot(st.xy,vec2(0.5,0.5));
    //return fract(dot(st.xy,vec2(1.60,0.2))*3.0);
}
*/

//vec2 st = gl_FragCoord.xy/(512);
    //st *= 3.0 * 170; // Scale the coordinate system by 10
    //st *= 9.0;
    //st *= 3.0;
    //vec2 ipos = floor(st);  // get the integer coords
    //st = fract(st);
    // Assign a random value based on the integer coord
    //vec3 color = vec3(random( ipos ));
    //float randomu = rand(vec2(0.5,0.5));
    //float randomu = rn(9.0);
    //if (randomu>0.5){
        //ipos = vec2(ipos.x+0.5,ipos.y+0.5);
    //}
    //ipos = vec2(ipos.x+randomu,ipos.y+randomu);
    
/*
    for (int i=0; i<3; i++){

        if(mod(ipos.x,3) == i){
            if(mod(ipos.y,3) == 0.0){
                if(rand(gl_FragCoord.xy)<0.5) fColor = gray;
                else fColor = black;
                break;
            }
            if(mod(ipos.y,3) == 1.0){
                if(rand(gl_FragCoord.xy)<0.5)fColor = gray;
                else fColor = black;
                break;
            }
            if(mod(ipos.y,3) == 2.0){
                if(rand(gl_FragCoord.xy)<0.5)fColor = gray;
                else fColor = black;
                break;

            }
        }

    }
*/




    
    
 
