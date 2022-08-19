/*
--------------------------------------------------
    James William Fletcher (github.com/mrbid)
--------------------------------------------------
    Auto-shoot / trigger bot for CS:GO.
    https://github.com/tfcnn
    	July 2021

    Compile:
    clang aimbot.c -Ofast -lX11 -lm -o aim
*/

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <sys/time.h>

#pragma GCC diagnostic ignored "-Wgnu-folding-constant"

#define uint unsigned int

#define SCAN_VARIANCE 0
#define SCAN_DELAY 1000 
#define ACTIVATION_SENITIVITY 0.75
#define REPEAT_ACTIVATION 2
#define FIRE_RATE_LIMIT_MS 800
#define TRIGGER_MULTIPLIER 2

#define SCAN_WIDTH 28
#define SCAN_HEIGHT 28
const uint sw = SCAN_WIDTH;
const uint sh = SCAN_HEIGHT;
const uint sw2 = sw/2;
const uint sh2 = sh/2;
const uint slc = sw*sh;
const uint slall = slc*3;

uint x=0, y=0;

float input[slall] = {0};
    double r[slc] = {0};
    double g[slc] = {0};
    double b[slc] = {0};

Display *d;
int si;
Window twin;
GC gc = 0;
int tc = 0;

/***************************************************
   ~~ Neural Network Forward-Pass
*/
float processModel(const float* input)
{
    // write input to file
    FILE *f = fopen("input.dat", "wb");
    if(f != NULL)
    {
        const size_t wbs = slall * sizeof(float);
        if(fwrite(input, 1, wbs, f) != wbs)
            return 0;
        fclose(f);
    }

    // load last result
    float ret = 0;
    f = fopen("r.dat", "rb");
    if(f != NULL)
    {
        if(fread(&ret, 1, sizeof(float), f) != sizeof(float))
            return 0;
        fclose(f);
    }

    // return
    return ret;
}


/***************************************************
   ~~ Utils
*/
uint64_t microtime()
{
	struct timeval tv;
	struct timezone tz;
	memset(&tz, 0, sizeof(struct timezone));
	gettimeofday(&tv, &tz);
	return 1000000 * tv.tv_sec + tv.tv_usec;
}

uint qRand(const uint min, const uint max)
{
    static float rndmax = (float)RAND_MAX;
    return ( ( (((float)rand())+1e-7) / rndmax ) * ((max+1)-min) ) + min;
}

//https://www.cl.cam.ac.uk/~mgk25/ucs/keysymdef.h
int key_is_pressed(KeySym ks)
{
    Display *dpy = XOpenDisplay(":0");
    char keys_return[32];
    XQueryKeymap(dpy, keys_return);
    KeyCode kc2 = XKeysymToKeycode(dpy, ks);
    int isPressed = !!(keys_return[kc2 >> 3] & (1 << (kc2 & 7)));
    XCloseDisplay(dpy);
    return isPressed;
}

void speakS(const char* text)
{
    char s[256];
    sprintf(s, "/usr/bin/espeak \"%s\"", text);
    if(system(s) <= 0)
        sleep(1);
}

void speakI(const int i)
{
    char s[256];
    sprintf(s, "/usr/bin/espeak %i", i);
    if(system(s) <= 0)
        sleep(1);
}

void speakF(const double f)
{
    char s[256];
    sprintf(s, "/usr/bin/espeak %.1f", f);
    if(system(s) <= 0)
        sleep(1);
}

Window getWindow() // gets child window mouse is over
{
    Display *d = XOpenDisplay((char *) NULL);
    if(d == NULL)
        return 0;
    int si = XDefaultScreen(d);
    XEvent event;
    memset(&event, 0x00, sizeof(event));
    XQueryPointer(d, RootWindow(d, si), &event.xbutton.root, &event.xbutton.window, &event.xbutton.x_root, &event.xbutton.y_root, &event.xbutton.x, &event.xbutton.y, &event.xbutton.state);
    event.xbutton.subwindow = event.xbutton.window;
    while(event.xbutton.subwindow)
    {
        event.xbutton.window = event.xbutton.subwindow;
        XQueryPointer(d, event.xbutton.window, &event.xbutton.root, &event.xbutton.subwindow, &event.xbutton.x_root, &event.xbutton.y_root, &event.xbutton.x, &event.xbutton.y, &event.xbutton.state);
    }
    const Window ret = event.xbutton.window;
    XCloseDisplay(d);
    return ret;
}

void processScanArea(Window w)
{
    // get image block
    XImage *img;
    if(SCAN_VARIANCE == 0)
        img = XGetImage(d, w, x-sw2, y-sh2, sw, sh, AllPlanes, XYPixmap);
    else
        img = XGetImage(d, w, (x+qRand(-SCAN_VARIANCE, SCAN_VARIANCE))-sw2, (y+qRand(-SCAN_VARIANCE, SCAN_VARIANCE))-sh2, sw, sh, AllPlanes, XYPixmap);
    if(img == NULL)
        return;

    // colour map
    const Colormap map = XDefaultColormap(d, si);

    // extract colour information
    double rh = 0, rl = 99999999999999, rm = 0;
    double gh = 0, gl = 99999999999999, gm = 0;
    double bh = 0, bl = 99999999999999, bm = 0;
    int i = 0;
    for(int y = 0; y < sh; y++)
    {
        for(int x = 0; x < sw; x++)
        {
            XColor c;
            c.pixel = XGetPixel(img, x, y);
            XQueryColor(d, map, &c);

            r[i] = (double)c.red;
            g[i] = (double)c.green;
            b[i] = (double)c.blue;

            if(r[i] > rh){rh = r[i];}
            if(r[i] < rl){rl = r[i];}
            rm += r[i];

            if(g[i] > gh){gh = g[i];}
            if(g[i] < gl){gl = g[i];}
            gm += g[i];

            if(b[i] > bh){bh = b[i];}
            if(b[i] < bl){bl = b[i];}
            bm += b[i];

            i++;
        }
    }

    // free image block
    XFree(img);

    /////////////////
    // mean normalised

    rm /= slc;
    gm /= slc;
    bm /= slc;

    const double rmd = rh-rl;
    const double gmd = gh-gl;
    const double bmd = bh-bl;

    for(uint i = 0, i2 = 0; i < slall; i += 3, i2++)
    {
        input[i]   = ((r[i2]-rm)+1e-7) / (rmd+1e-7);
        input[i+1] = ((g[i2]-gm)+1e-7) / (gmd+1e-7);
        input[i+2] = ((b[i2]-bm)+1e-7) / (bmd+1e-7);
    }
}

/***************************************************
   ~~ Program Entry Point
*/
int main(int argc, char *argv[])
{
    printf("James William Fletcher (james@voxdsp.com)\n\n");
    printf("Hotkeys:\n");
    printf("L-CTRL + L-ALT = Toggle BOT ON/OFF\n");
    printf("R-CTRL + R-ALT = Toggle HOTKEYS ON/OFF\n");
    printf("\nL-SHIFT / V = Autoshoot while pressed.\n\n");
    printf("T = Toggle autoshoot on/off.\n");
    printf("P = Toggle crosshair.\n");
    printf("C = Output input array from reticule area.\n");
    printf("G = Get activation for reticule area.\n");
    printf("H = Get scans per second.\n");
    printf("\n\n");

    //
    
    XEvent event;
    memset(&event, 0x00, sizeof(event));
    
    uint enable = 0;
    uint autofire = 0;
    uint crosshair = 0;
    uint hotkeys = 1;

    //
    
    while(1)
    {
        // loop every SCAN_DELAY ms (1,000 microsecond = 1 millisecond)
        usleep(SCAN_DELAY);

        // bot toggle
        if(key_is_pressed(XK_Control_L) && key_is_pressed(XK_Alt_L))
        {
            if(enable == 0)
            {
                // open display 0
                d = XOpenDisplay((char *) NULL);
                if(d == NULL)
                    continue;

                // get default screen
                si = XDefaultScreen(d);

                // get graphics context
                gc = DefaultGC(d, si);

                // get window
                twin = getWindow();

                // get center window point (x & y)
                XWindowAttributes attr;
                XGetWindowAttributes(d, twin, &attr);
                x = attr.width/2;
                y = attr.height/2;

                // set mouse event
                memset(&event, 0x00, sizeof(event));
                event.type = ButtonPress;
                event.xbutton.button = Button1;
                event.xbutton.same_screen = True;
                event.xbutton.subwindow = twin;
                event.xbutton.window = twin;

                enable = 1;
                usleep(300000);
                printf("BOT: ON [%ix%i]\n", x, y);
                speakS("on");
            }
            else
            {
                enable = 0;
                usleep(300000);
                XCloseDisplay(d);
                printf("BOT: OFF\n");
                speakS("off");
            }
        }
        
        // toggle bot on/off
        if(enable == 1 && getWindow() == twin)
        {
            // input toggle
            if(key_is_pressed(XK_Control_R) && key_is_pressed(XK_Alt_R))
            {
                if(hotkeys == 0)
                {
                    hotkeys = 1;
                    usleep(300000);
                    printf("HOTKEYS: ON [%ix%i]\n", x, y);
                    speakS("hk on");
                }
                else
                {
                    hotkeys = 0;
                    usleep(300000);
                    printf("HOTKEYS: OFF\n");
                    speakS("hk off");
                }
            }

            if(hotkeys == 1)
            {
                // detect when pressed
                if(autofire <= 1)
                {
                    if(key_is_pressed(XK_Shift_L))
                    {
                        autofire = 1;
                    }
                    else
                    {
                        if(key_is_pressed(XK_V))
                        {
                            autofire = 1;
                        }
                        else
                        {
                            autofire = 0;
                            remove("r.dat");
                        }
                    }
                }

                // autofire toggle
                if(key_is_pressed(XK_T))
                {
                    if(autofire == 0)
                    {
                        autofire = 2;
                        usleep(300000);
                        printf("AUTOFIRE: ON\n");
                        speakS("af on");
                    }
                    else
                    {
                        autofire = 0;
                        remove("r.dat");
                        usleep(300000);
                        printf("AUTOFIRE: OFF\n");
                        speakS("af off");
                    }
                }
                
                // crosshair toggle
                if(key_is_pressed(XK_P))
                {
                    if(crosshair == 0)
                    {
                        crosshair = 1;
                        usleep(300000);
                        printf("CROSSHAIR: ON\n");
                        speakS("cx on");
                    }
                    else
                    {
                        crosshair = 0;
                        usleep(300000);
                        printf("CROSSHAIR: OFF\n");
                        speakS("cx off");
                    }
                }

                // print input data
                if(key_is_pressed(XK_C))
                {
                    processScanArea(twin);

                    // per channel
                    printf("R: ");
                    for(uint i = 0; i < slc; i++)
                        printf("%.2f ", r[i]);
                    printf("\n\n");

                    printf("G: ");
                    for(uint i = 0; i < slc; i++)
                        printf("%.2f ", g[i]);
                    printf("\n\n");

                    printf("B: ");
                    for(uint i = 0; i < slc; i++)
                        printf("%.2f ", b[i]);
                    printf("\n\n");
                }

            }
            
            if(autofire > 0) // left mouse trigger on activation
            {
                processScanArea(twin);
                const float activation = processModel(&input[0]);

                // passed minimum activation?
                if(activation > ACTIVATION_SENITIVITY)
                {
                    tc++;

                    // did we activate enough times in a row to be sure this is a target?
                    if(tc > REPEAT_ACTIVATION)
                    {
                        // fire off as many shots as we need to
                        for(int i = 0; i < TRIGGER_MULTIPLIER; i++)
                        {
                            // fire mouse down
                            event.type = ButtonPress;
                            event.xbutton.state = 0;
                            XSendEvent(d, PointerWindow, True, 0xfff, &event);
                            XFlush(d);
                            
                            // wait 100ms (or ban for suspected cheating)
                            usleep(100000);
                            
                            // release mouse down
                            event.type = ButtonRelease;
                            event.xbutton.state = 0x100;
                            XSendEvent(d, PointerWindow, True, 0xfff, &event);
                            XFlush(d);

                            // fire limit
                            usleep(FIRE_RATE_LIMIT_MS * 100);
                        }

                        // fire limit
                        usleep(FIRE_RATE_LIMIT_MS * 1000);
                    }
                }
                else
                {
                    tc = 0;
                }
            }
            else if(hotkeys == 1 && key_is_pressed(XK_G)) // print activation when pressed
            {
                processScanArea(twin);
                const float ret = processModel(&input[0]);
                if(ret > ACTIVATION_SENITIVITY)
                {
                    printf("\e[93mA: %f\e[0m\n", ret);
                    XSetForeground(d, gc, 65280);
                    XDrawRectangle(d, event.xbutton.window, gc, x-sw2-1, y-sh2-1, sw+2, sh+2);
                    XSetForeground(d, gc, 0);
                    XDrawRectangle(d, event.xbutton.window, gc, x-sw2-2, y-sh2-2, sw+4, sh+4);
                    XFlush(d);
                }
                else
                {
                    printf("A: %f\n", ret);
                    XSetForeground(d, gc, 16711680);
                    XDrawRectangle(d, event.xbutton.window, gc, x-sw2-1, y-sh2-1, sw+2, sh+2);
                    XSetForeground(d, gc, 0);
                    XDrawRectangle(d, event.xbutton.window, gc, x-sw2-2, y-sh2-2, sw+4, sh+4);
                    XFlush(d);
                }                
            }
            else if(hotkeys == 1 && key_is_pressed(XK_H)) // print samples per second when pressed
            {
                static uint64_t st = 0;
                static uint sc = 0;
                processScanArea(twin);
                const float ret = processModel(&input[0]);
                sc++;
                if(microtime() - st >= 1000000)
                {
                    printf("\e[36mSPS: %u\e[0m\n", sc);
                    sc = 0;
                    st = microtime();
                }              
            }

            if(crosshair == 1)
            {
                XSetForeground(d, gc, 0);
                XDrawPoint(d, event.xbutton.window, gc, x, y);
                XSetForeground(d, gc, 65280);
                XDrawRectangle(d, event.xbutton.window, gc, x-1, y-1, 2, 2);
                XSetForeground(d, gc, 0);
                XDrawRectangle(d, event.xbutton.window, gc, x-2, y-2, 4, 4);
                XFlush(d);
            }

        ///
        }
    }

    // done, never gets here in regular execution flow
    return 0;
}
