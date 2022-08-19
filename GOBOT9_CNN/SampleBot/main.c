/*
--------------------------------------------------
    James William Fletcher (github.com/mrbid)
--------------------------------------------------
    BMP Dataset collector hard-coded to 96x192.

    Designed for use in CS:GO suggested settings:

    sv_cheats 1
    hud_showtargetid = 0
    cl_teamid_overhead_mode = 1
    cl_teamid_overhead_maxdist 0.1
    bot_stop 1

    Turn off the crosshair or turn it into a single
    dot with no border.

    Compile:
    clang main.c -Ofast -lX11 -lm -o sbot
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

#pragma GCC diagnostic ignored "-Wgnu-folding-constant"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define uint unsigned int
#define SCAN_WIDTH 28
#define SCAN_HEIGHT 28

const uint sw = SCAN_WIDTH;
const uint sh = SCAN_HEIGHT;
const uint sw2 = sw/2;
const uint sh2 = sh/2;
const uint slc = sw*sh;
const uint slall = slc*3;

uint x=0, y=0;
Display *d;
int si;
Window twin;
GC gc = 0;

/***************************************************
   ~~ Utils
*/
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

void speakSS(const char* text)
{
    char s[256];
    sprintf(s, "/usr/bin/espeak -s 360 \"%s\"", text);
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

void saveSample(Window w, const char* name)
{
    // get image block
    XImage *img = XGetImage(d, w, x-sw2, y-sh2, sw, sh, AllPlanes, XYPixmap);
    if(img == NULL)
        return;

    // colour map
    const Colormap map = XDefaultColormap(d, si);

    // extract colour information
    double r[slc] = {0};
    double g[slc] = {0};
    double b[slc] = {0};
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

            i++;
        }
    }

    // free image block
    XFree(img);

    /////////////////
    // regular 0-255 byte per colour channel
    char rgbbytes[slall] = {0};
    for(uint i = 0, i2 = 0; i < sizeof(rgbbytes); i += 3, i2++)
    {
        rgbbytes[i]   = (char)((r[i2] / 65535.0) * 255);
        rgbbytes[i+1] = (char)((g[i2] / 65535.0) * 255);
        rgbbytes[i+2] = (char)((b[i2] / 65535.0) * 255);
    }

    // save to file
    stbi_write_bmp(name, sw, sh, 3, &rgbbytes);
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
    printf("P = Toggle crosshair\n");
    printf("V = Show sample frame area\n");
    printf("E = Sample enemy to dataset.\n");
    printf("Q = Sample non-enemy to dataset.\n");
    printf("\n\n");

    //
    srand(time(0));

    XEvent event;
    memset(&event, 0x00, sizeof(event));
    
    uint enable = 0;
    uint crosshair = 0;
    uint hotkeys = 1;

    //
    
    while(1)
    {
        // loop every 10 ms (1,000 microsecond = 1 millisecond)
        usleep(1000);

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
                        printf("Don't have this enabled while taking samples, the crosshair will be burned into your training data.\n");
                    }
                }
            }
            
            if(hotkeys == 1 && key_is_pressed(XK_V))
            {
                // draw sample outline
                XSetForeground(d, gc, 0);
                XDrawRectangle(d, event.xbutton.window, gc, x-sw2-1, y-sh2-1, sw+2, sh+2);
                XSetForeground(d, gc, 0);
                XDrawRectangle(d, event.xbutton.window, gc, x-sw2-2, y-sh2-2, sw+4, sh+4);
                XFlush(d);
            }
            else if(hotkeys == 1 && key_is_pressed(XK_E))
            {
                char name[32];
                sprintf(name, "target/%i.bmp", rand());
                saveSample(twin, name);

                // draw sample outline
                XSetForeground(d, gc, 65280);
                XDrawRectangle(d, event.xbutton.window, gc, x-sw2-1, y-sh2-1, sw+2, sh+2);
                XSetForeground(d, gc, 0);
                XDrawRectangle(d, event.xbutton.window, gc, x-sw2-2, y-sh2-2, sw+4, sh+4);
                XFlush(d);

                speakSS("t");
            }
            else if(hotkeys == 1 && key_is_pressed(XK_Q))
            {
                char name[32];
                sprintf(name, "nontarget/%i.bmp", rand());
                saveSample(twin, name);

                // draw sample outline
                XSetForeground(d, gc, 16711680);
                XDrawRectangle(d, event.xbutton.window, gc, x-sw2-1, y-sh2-1, sw+2, sh+2);
                XSetForeground(d, gc, 0);
                XDrawRectangle(d, event.xbutton.window, gc, x-sw2-2, y-sh2-2, sw+4, sh+4);
                XFlush(d);

                usleep(350000);
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
