/*
--------------------------------------------------
    James William Fletcher (github.com/mrbid)
        October 2021
--------------------------------------------------

    CS:GO - CNN 28x28 - DATASET V3
    
    Prereq:
    sudo apt install libxdo-dev libxdo3 libespeak1 libespeak-dev espeak

    Compile:
    clang main.c -Ofast -mavx -mfma -lX11 -lxdo -lespeak -lm -o aim
*/

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <X11/Xutil.h>
#include <signal.h>
#include <sys/stat.h>

#include <xdo.h>
#include <espeak/speak_lib.h>

#pragma GCC diagnostic ignored "-Wgnu-folding-constant"
#pragma GCC diagnostic ignored "-Wunused-result"

#define SCAN_DELAY 1000
#define ACTIVATION_SENITIVITY 0.9f
#define REPEAT_ACTIVATION 0

#define uint unsigned int
#define SCAN_WIDTH 28
#define SCAN_HEIGHT 28

const uint sw = SCAN_WIDTH;
const uint sh = SCAN_HEIGHT;
const uint sw2 = sw/2;
const uint sh2 = sh/2;
const uint slc = sw*sh;
const uint slall = slc*3;

float input[slall] = {0};

Display *d;
int si;
Window twin;
unsigned int x=0, y=0;
unsigned int tc = 0;

char targets_dir[256];
char nontargets_dir[256];


/***************************************************
   ~~ Neural Network Forward-Pass
*/
float processModel(const float* input)
{
    // write input to file
    FILE *f = fopen("/dev/shm/pred_input.dat", "wb");
    if(f != NULL)
    {
        const size_t wbs = slall * sizeof(float);
        if(fwrite(input, 1, wbs, f) != wbs)
            return 0;
        fclose(f);
    }

    // load last result
    float ret = 0;
    f = fopen("/dev/shm/pred_r.dat", "rb");
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
//https://www.cl.cam.ac.uk/~mgk25/ucs/keysymdef.h
//https://stackoverflow.com/questions/18281412/check-keypress-in-c-on-linux/52801588
int key_is_pressed(Display* dpy, KeySym ks)
{
    char keys_return[32];
    XQueryKeymap(dpy, keys_return);
    KeyCode kc2 = XKeysymToKeycode(dpy, ks);
    int isPressed = !!(keys_return[kc2 >> 3] & (1 << (kc2 & 7)));
    return isPressed;
}

unsigned int espeak_fail = 0;
void speakS(const char* text)
{
    if(espeak_fail == 1)
    {
        char s[256];
        sprintf(s, "/usr/bin/espeak \"%s\"", text);
        system(s);
        usleep(33000);
    }
    else
    {
        espeak_Synth(text, strlen(text), 0, 0, 0, espeakCHARS_AUTO,NULL,NULL);
    }
}

uint qRand(const uint min, const uint max)
{
    static float rndmax = (float)RAND_MAX;
    return ( ( (((float)rand())+1e-7f) / rndmax ) * ((max+1)-min) ) + min;
}

uint64_t microtime()
{
    struct timeval tv;
    struct timezone tz;
    memset(&tz, 0, sizeof(struct timezone));
    gettimeofday(&tv, &tz);
    return 1000000 * tv.tv_sec + tv.tv_usec;
}

/***************************************************
   ~~ X11 Utils
*/

Window getWindow(Display* d, const int si) // gets child window mouse is over
{
    XEvent event;
    memset(&event, 0x00, sizeof(event));
    XQueryPointer(d, RootWindow(d, si), &event.xbutton.root, &event.xbutton.window, &event.xbutton.x_root, &event.xbutton.y_root, &event.xbutton.x, &event.xbutton.y, &event.xbutton.state);
    event.xbutton.subwindow = event.xbutton.window;
    while(event.xbutton.subwindow)
    {
        event.xbutton.window = event.xbutton.subwindow;
        XQueryPointer(d, event.xbutton.window, &event.xbutton.root, &event.xbutton.subwindow, &event.xbutton.x_root, &event.xbutton.y_root, &event.xbutton.x, &event.xbutton.y, &event.xbutton.state);
    }
    return event.xbutton.window;
}

Window findWindow(Display *d, Window current, char const *needle)
{
    Window ret = 0, root, parent, *children;
    unsigned cc;
    char *name = NULL;

    if(current == 0)
        current = XDefaultRootWindow(d);

    if(XFetchName(d, current, &name) > 0)
    {
        if(strstr(name, needle) != NULL)
        {
            XFree(name);
            return current;
        }
        XFree(name);
    }

    if(XQueryTree(d, current, &root, &parent, &children, &cc) != 0)
    {
        for(unsigned int i = 0; i < cc; ++i)
        {
            Window win = findWindow(d, children[i], needle);

            if(win != 0)
            {
                ret = win;
                break;
            }
        }
        XFree(children);
    }
    return ret;
}

void writePPM(const char* file, const unsigned char* data)
{
    FILE* f = fopen(file, "wb");
    if(f != NULL)
    {
        fprintf(f, "P6 28 28 255 ");
        fwrite(data, 1, slall, f);
        fclose(f);
    }
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
    unsigned char rgbbytes[slall] = {0};
    int i = 0;
    for(int y = 0; y < sh; y++)
    {
        for(int x = 0; x < sw; x++)
        {
            XColor c;
            c.pixel = XGetPixel(img, x, y);
            XQueryColor(d, map, &c);

            // 0-1 norm
            rgbbytes[i]   = (unsigned char)((((float)c.red+1e-7f)   / 65535.0f) * 255.0f)+0.5f;
            rgbbytes[++i] = (unsigned char)((((float)c.green+1e-7f) / 65535.0f) * 255.0f)+0.5f;
            rgbbytes[++i] = (unsigned char)((((float)c.blue+1e-7f)  / 65535.0f) * 255.0f)+0.5f;
            i++;
        }
    }

    // free image block
    XFree(img);

    // save to file
    writePPM(name, &rgbbytes[0]);
}

void processScanArea(Window w)
{
    // get image block
    XImage *img = XGetImage(d, w, x-sw2, y-sh2, sw, sh, AllPlanes, XYPixmap);
    if(img == NULL)
        return;

    // colour map
    const Colormap map = XDefaultColormap(d, si);

    // extract colour information
    int i = 0;
    for(int y = 0; y < sh; y++)
    {
        for(int x = 0; x < sw; x++)
        {
            XColor c;
            c.pixel = XGetPixel(img, x, y);
            XQueryColor(d, map, &c);

            // 0-1 norm
            input[i]   = c.red   / 65535.f;
            input[i+1] = c.green / 65535.f;
            input[i+2] = c.blue  / 65535.f;
            i += 3;
        }
    }

    // free image block
    XFree(img);
}

/***************************************************
   ~~ Console Utils
*/

int gre()
{
    int r = 0;
    while(r == 0 || r == 15 || r == 16 || r == 189)
    {
        r = (rand()%229)+1;
    }
    return r;
}
void random_printf(const char* text)
{
    const unsigned int len = strlen(text);
    for(unsigned int i = 0; i < len; i++)
    {
        printf("\e[38;5;%im", gre());
        printf("%c", text[i]);
    }
    printf("\e[38;5;123m");
}

void rainbow_printf(const char* text)
{
    static unsigned int base_clr = 0;
    if(base_clr == 0)
        base_clr = (rand()%125)+55;
    
    base_clr += 3;

    unsigned int clr = base_clr;
    const unsigned int len = strlen(text);
    for(unsigned int i = 0; i < len; i++)
    {
        clr++;
        printf("\e[38;5;%im", clr);
        printf("%c", text[i]);
    }
    printf("\e[38;5;123m");
}

void rainbow_line_printf(const char* text)
{
    static unsigned int base_clr = 0;
    if(base_clr == 0)
        base_clr = (rand()%125)+55;
    
    printf("\e[38;5;%im", base_clr);
    base_clr++;
    if(base_clr >= 230)
        base_clr = (rand()%125)+55;

    const unsigned int len = strlen(text);
    for(unsigned int i = 0; i < len; i++)
        printf("%c", text[i]);
    printf("\e[38;5;123m");
}

/***************************************************
   ~~ Program Entry Point
*/
int main()
{
    srand(time(0));
    signal(SIGPIPE, SIG_IGN);

    if(espeak_Initialize(AUDIO_OUTPUT_SYNCH_PLAYBACK, 0, 0, 0) < 0)
        espeak_fail = 1;

    printf("\033[H\033[J");
    rainbow_printf("James William Fletcher (james@voxdsp.com)\n\n");
    rainbow_printf("L-CTRL + L-ALT = Toggle BOT ON/OFF\n");
    rainbow_printf("R-CTRL + R-ALT = Toggle HOTKEYS ON/OFF\n");
    rainbow_printf("P = Toggle crosshair\n\n");
    rainbow_printf("L = Toggle sample capture\n");
    rainbow_printf("Q = Capture non-target sample\n\n");
    rainbow_printf("G = Get activation for reticule area.\n");
    rainbow_printf("H = Get scans per second.\n");
    rainbow_printf("\nDisable the game crosshair or make the crosshair a single green pixel, or if your monitor provides a crosshair use that.\n\n");
    rainbow_printf("This bot will only auto trigger when W,A,S,D are not being pressed. (so when your not moving in game, aka stationary)\n\n");

    xdo_t* xdo;
    XColor c[9];
    GC gc = 0;
    unsigned int enable = 0;
    unsigned int offset = 3;
    unsigned int crosshair = 0;
    unsigned int sample_capture = 0;
    unsigned int hotkeys = 1;
    unsigned int draw_sa = 0;
    time_t ct = time(0);

    // open display 0
    d = XOpenDisplay(":0");
    if(d == NULL)
    {
        printf("Failed to open display\n");
        return 0;
    }

    // get default screen
    si = XDefaultScreen(d);

    // find bottom window
    twin = findWindow(d, 0, "Counter-Strike");
    if(twin != 0)
        printf("CS:GO Win: 0x%lX\n", twin);

    //xdo
    xdo = xdo_new(":0.0");

    // set console title
    Window awin;
    xdo_get_active_window(xdo, &awin);
    xdo_set_window_property(xdo, awin, "WM_NAME", "CS:GO - CNN 28x28 - DATASET V3");

    // get graphics context
    gc = DefaultGC(d, si);
    
    while(1)
    {
        // loop every 1 ms (1,000 microsecond = 1 millisecond)
        usleep(SCAN_DELAY);

        // inputs
        if(key_is_pressed(d, XK_Control_L) && key_is_pressed(d, XK_Alt_L))
        {
            if(enable == 0)
            {                
                // get window
                //xdo_get_active_window(xdo, &twin);
                //twin = getWindow(d, si);
                twin = findWindow(d, 0, "Counter-Strike");

                // get center window point (x & y)
                XWindowAttributes attr;
                XGetWindowAttributes(d, twin, &attr);
                x = attr.width/2;
                y = attr.height/2;

                // toggle
                enable = 1;
                usleep(300000);
                rainbow_line_printf("BOT: ON ");
                printf("[%ix%i]\n", x, y);
                speakS("on");
            }
            else
            {
                enable = 0;
                usleep(300000);
                rainbow_line_printf("BOT: OFF\n");
                speakS("off");
            }
        }
        
        // bot on/off
        if(enable == 1)
        {
            // input toggle
            if(key_is_pressed(d, XK_Control_R) && key_is_pressed(d, XK_Alt_R))
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
                    rainbow_line_printf("HOTKEYS: OFF\n");
                    speakS("hk off");
                }
            }

            if(hotkeys == 1)
            {
                // crosshair toggle
                if(key_is_pressed(d, XK_P))
                {
                    if(crosshair == 0)
                    {
                        crosshair = 1;
                        usleep(300000);
                        rainbow_line_printf("CROSSHAIR: ON\n");
                        speakS("cx on");
                    }
                    else
                    {
                        crosshair = 0;
                        usleep(300000);
                        rainbow_line_printf("CROSSHAIR: OFF\n");
                        speakS("cx off");
                    }
                }

                static uint64_t scd = 0;
                if(key_is_pressed(d, XK_Q) && sample_capture == 1 && microtime() > scd)
                {
                    char name[32];
                    sprintf(name, "%s/%i.ppm", nontargets_dir, rand());
                    saveSample(twin, name);

                    draw_sa = 100;

                    scd = microtime() + 350000;
                }

                // sample capture toggle
                if(key_is_pressed(d, XK_L))
                {
                    if(sample_capture == 0)
                    {
                        char* home = getenv("HOME");
                        sprintf(targets_dir, "%s/Desktop/targets", home);
                        mkdir(targets_dir, 0777);
                        sprintf(nontargets_dir, "%s/Desktop/nontargets", home);
                        mkdir(nontargets_dir, 0777);

                        sample_capture = 1;
                        usleep(300000);
                        rainbow_line_printf("SAMPLE CAPTURE: ON\n");
                        speakS("sc on");
                    }
                    else
                    {
                        sample_capture = 0;
                        usleep(300000);
                        rainbow_line_printf("SAMPLE CAPTURE: OFF\n");
                        speakS("sc off");
                    }
                }

                if(key_is_pressed(d, XK_G)) // print activation when pressed
                {
                    processScanArea(twin);
                    const float ret = processModel(&input[0]);
                    if(ret >= ACTIVATION_SENITIVITY)
                    {
                        printf("\e[93mA: %f\e[0m\n", ret);
                        XSetForeground(d, gc, 65280);
                        XDrawRectangle(d, twin, gc, x-sw2-1, y-sh2-1, sw+2, sh+2);
                        XSetForeground(d, gc, 0);
                        XDrawRectangle(d, twin, gc, x-sw2-2, y-sh2-2, sw+4, sh+4);
                        XFlush(d);
                    }
                    else
                    {
                        printf("A: %f\n", ret);
                        XSetForeground(d, gc, 16711680);
                        XDrawRectangle(d, twin, gc, x-sw2-1, y-sh2-1, sw+2, sh+2);
                        XSetForeground(d, gc, 0);
                        XDrawRectangle(d, twin, gc, x-sw2-2, y-sh2-2, sw+4, sh+4);
                        XFlush(d);
                    }
                }

                if(key_is_pressed(d, XK_H)) // print samples per second when pressed
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
            }

            static uint64_t rft = 0;
            if(key_is_pressed(d, XK_W) == 0 && key_is_pressed(d, XK_A) == 0 && key_is_pressed(d, XK_S) == 0 && key_is_pressed(d, XK_D) == 0 && microtime() > rft)
            {
                processScanArea(twin);
                const float activation = processModel(&input[0]);
                if(activation >= ACTIVATION_SENITIVITY)
                {
                    tc++;

                    // did we activate enough times in a row to be sure this is a target?
                    if(tc > REPEAT_ACTIVATION)
                    {
                        if(sample_capture == 1)
                        {
                            char name[32];
                            sprintf(name, "%s/%i.ppm", targets_dir, rand());
                            saveSample(twin, name);
                        }

                        xdo_mouse_down(xdo, CURRENTWINDOW, 1);
                        usleep(100000);
                        xdo_mouse_up(xdo, CURRENTWINDOW, 1);

                        if(sample_capture == 1)
                            rft = microtime() + 350000;
                        else
                            rft = microtime() + 80000;

                        // display ~1s recharge time
                        if(crosshair != 0)
                        {
                            crosshair = 2;
                            ct = time(0);
                        }
                    }
                }
                else
                {
                    tc = 0;
                }
            }

            if(crosshair == 1)
            {
                if(sample_capture == 1)
                {
                    if(draw_sa > 0)
                    {
                        // draw sample outline
                        XSetForeground(d, gc, 16711680);
                        XDrawRectangle(d, twin, gc, x-sw2-1, y-sh2-1, sw+2, sh+2);
                        XSetForeground(d, gc, 16711680);
                        XDrawRectangle(d, twin, gc, x-sw2-2, y-sh2-2, sw+4, sh+4);
                        XFlush(d);
                        draw_sa -= 1;
                    }
                    else
                    {
                        XSetForeground(d, gc, 16777215);
                        XDrawRectangle(d, twin, gc, x-sw2-1, y-sh2-1, sw+2, sh+2);
                        XSetForeground(d, gc, 16776960);
                        XDrawRectangle(d, twin, gc, x-sw2-2, y-sh2-2, sw+4, sh+4);
                        XFlush(d);
                    }
                }
                else
                {
                    XSetForeground(d, gc, 65280);
                    XDrawRectangle(d, twin, gc, x-sw2-1, y-sh2-1, sw+2, sh+2);
                    XFlush(d);
                }
            }

            if(crosshair == 2)
            {
                if(time(0) > ct+1)
                    crosshair = 1;

                if(sample_capture == 1)
                {
                    XSetForeground(d, gc, 16777215);
                    XDrawRectangle(d, twin, gc, x-sw2-1, y-sh2-1, sw+2, sh+2);
                    XSetForeground(d, gc, 16711680);
                    XDrawRectangle(d, twin, gc, x-sw2-2, y-sh2-2, sw+4, sh+4);
                    XFlush(d);
                }
                else
                {
                    XSetForeground(d, gc, 16711680);
                    XDrawRectangle(d, twin, gc, x-sw2-1, y-sh2-1, sw+2, sh+2);
                    XSetForeground(d, gc, 65280);
                    XDrawRectangle(d, twin, gc, x-sw2-2, y-sh2-2, sw+4, sh+4);
                    XFlush(d);
                }
            }
        }

        //
    }

    // done, never gets here in regular execution flow
    XCloseDisplay(d);
    return 0;
}

