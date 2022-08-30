/*
--------------------------------------------------
    James William Fletcher (github.com/mrbid)
        AUGUST 2022
--------------------------------------------------
    
    You may want to install espeak via your package manager.

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <unistd.h>
#include <stdint.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <sys/time.h>
#include <sys/stat.h>

// general
#pragma GCC diagnostic ignored "-Wgnu-folding-constant"
#define uint unsigned int

// scan area vars
#define SCAN_AREA 28
const uint r0 = SCAN_AREA;  // dimensions of sample image square
const uint r2 = r0*r0;      // total pixels in square
const uint r2i = r2*3;      // total inputs to neural net pixels*channels
const uint rd2 = r0/2;      // total pixels divided by two
uint x=0, y=0;
float input[r2i] = {0};

// x11 vars
Display *d;
int si;
Window twin = 0;
GC gc = 0;

// sample capture vars
#define PRELOG_SAVESAMPLE 0 // set this to 0 to offset some scanning load to the saving operation
char targets_dir[256];
unsigned char rgbbytes[r2i] = {0};

// settings
uint enable = 0;
uint sample_capture = 0;
uint crosshair = 1;
uint hotkeys = 1;

// hyperparameters that you can change
#define TRIGGER 1                   // how many times to pull the trigger when a target is detected
#define SCAN_VARIANCE 0.f           // how much to randomly wiggle the scan area between scans
#define SCAN_DELAY 1000             // scan frequency delay in microseconds
#define ACTIVATION_SENITIVITY 0.9f  // minimum activation sensitivity to fire a shot
#define REPEAT_ACTIVATION 0         // how many positive activations in a row before firing a shot
#define FIRE_RATE_LIMIT_MS 300      // delay between firing shots in milliseconds
#define TRIGGER_RATE_LIMIT_MS 50    // delay between firing shots between TRIGGER iterations in milliseconds

/***************************************************
   ~~ Neural Network Forward-Pass
*/
uint64_t microtime();
float processModel(const float* input)
{
    // write input to file
    FILE *f = fopen("/dev/shm/pred_input.dat", "wb");
    if(f != NULL)
    {
        const size_t wbs = r2i * sizeof(float);
        if(fwrite(input, 1, wbs, f) != wbs)
            return 0.f;
        fclose(f);
    }

    // prevent old outputs returning
    static uint64_t stale = 0;
    if(microtime() - stale > 60000)
    {
        remove("/dev/shm/pred_r.dat");
        stale = microtime();
        return 0.f;
    }

    // load last result
    float ret = 0.f;
    f = fopen("/dev/shm/pred_r.dat", "rb");
    if(f != NULL)
    {
        if(fread(&ret, 1, sizeof(float), f) != sizeof(float))
            return 0.f;
        fclose(f);
    }

    // return
    stale = microtime();
    return ret;
}

/***************************************************
   ~~ Utils
*/
uint qRand(const float min, const float max)
{
    static float rndmax = 1.f/(float)RAND_MAX;
    return ( ( ( ((float)rand()) * rndmax ) * (max-min) ) + min ) + 0.5f;
}

void writePPM(const char* file, const unsigned char* data)
{
    FILE* f = fopen(file, "wb");
    if(f != NULL)
    {
        fprintf(f, "P6 28 28 255 ");
        fwrite(data, 1, r2i, f);
        fclose(f);
    }
}

#if PRELOG_SAVESAMPLE == 0
// this is an alternate efficient method if you don't like making
// two copies in memory on every processScanArea() call.
// we convert the existing float array back into a byte array.
void saveSample(const char* name)
{
    for(uint i = 0; i < r2i; i++)
        rgbbytes[i] = (unsigned char)(input[i] * 255.f);
    writePPM(name, &rgbbytes[0]);
}
#endif

uint64_t microtime()
{
    struct timeval tv;
    struct timezone tz;
    memset(&tz, 0, sizeof(struct timezone));
    gettimeofday(&tv, &tz);
    return 1000000 * tv.tv_sec + tv.tv_usec;
}

void rainbow_printf(const char* text)
{
    static unsigned char base_clr = 0;
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
    static unsigned char base_clr = 0;
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

//https://www.cl.cam.ac.uk/~mgk25/ucs/keysymdef.h
int key_is_pressed(KeySym ks)
{
    char keys_return[32];
    XQueryKeymap(d, keys_return);
    KeyCode kc2 = XKeysymToKeycode(d, ks);
    int isPressed = !!(keys_return[kc2 >> 3] & (1 << (kc2 & 7)));
    return isPressed;
}

void speakS(const char* text)
{
    char s[256];
    sprintf(s, "/usr/bin/espeak \"%s\"", text);
    if(system(s) <= 0)
        sleep(1);
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

void processScanArea(Window w)
{
    // get image block
    XImage *img;
#ifdef SCAN_VARIANCE
    img = XGetImage(d, w, (x+qRand(-SCAN_VARIANCE, SCAN_VARIANCE))-rd2, (y+qRand(-SCAN_VARIANCE, SCAN_VARIANCE))-rd2, r0, r0, AllPlanes, XYPixmap);
#else
    img = XGetImage(d, w, x-rd2, y-rd2, r0, r0, AllPlanes, XYPixmap);
#endif
    if(img == NULL)
        return;

    // colour map
    const Colormap map = XDefaultColormap(d, si);

    // extract colour information
    int i = 0;
    for(int y = 0; y < r0; y++)
    {
        for(int x = 0; x < r0; x++)
        {
            const unsigned long pixel = XGetPixel(img, x, y);
            const unsigned char sr = (pixel & img->red_mask) >> 16;
            const unsigned char sg = (pixel & img->green_mask) >> 8;
            const unsigned char sb = pixel & img->blue_mask;

#if PRELOG_SAVESAMPLE == 1
            rgbbytes[i]   = (unsigned char)sr;
            rgbbytes[i+1] = (unsigned char)sg;
            rgbbytes[i+2] = (unsigned char)sb;
#endif

            // 0-1 norm
            input[i]   = sr * 0.003921568859f; // / 255.f
            input[i+1] = sg * 0.003921568859f;
            input[i+2] = sb * 0.003921568859f;

            //
            i += 3;
        }
    }

    // free image block
    XFree(img);
}

void reprint()
{
    system("clear");
    rainbow_printf("James William Fletcher (github.com/mrbid)\n");
    rainbow_printf("L-CTRL + L-ALT = Toggle BOT ON/OFF\n");
    rainbow_printf("R-CTRL + R-ALT = Toggle HOTKEYS ON/OFF\n");
    rainbow_printf("P = Toggle crosshair.\n");
    rainbow_printf("L = Toggle sample capture.\n");
    rainbow_printf("E = Capture sample.\n");
    rainbow_printf("G = Get activation for reticule area.\n");
    rainbow_printf("H = Hold pressed to print scans per second.\n");
    printf("\e[38;5;76m");
    printf("\nMake the crosshair a single green pixel.\nOR disable the game crosshair and use the crosshair provided by this bot.\nOR if your monitor provides a crosshair use that. (this is best)\n\n");
    printf("This bot will only auto trigger when W,A,S,D & L-SHIFT are not being pressed.\n(so when your not moving in game, aka stationary)\n\nL-SHIFT allows you to disable the bot while stationary if desired.\n\n");
    printf("\e[38;5;123m");

    if(twin != 0)
    {
        printf("CS:GO Win: 0x%lX\n\n", twin);

        if(enable == 1)
            rainbow_line_printf("BOT: \033[1m\e[32mON\e[0m\n");
        else
            rainbow_line_printf("BOT: \033[1m\e[31mOFF\e[0m\n");

        if(hotkeys == 1)
            rainbow_line_printf("HOTKEYS: \033[1m\e[32mON\e[0m\n");
        else
            rainbow_line_printf("HOTKEYS: \033[1m\e[31mOFF\e[0m\n");

        if(sample_capture == 1)
            rainbow_line_printf("SAMPLE CAPTURE: \033[1m\e[32mON\e[0m\n");
        else
            rainbow_line_printf("SAMPLE CAPTURE: \033[1m\e[31mOFF\e[0m\n");

        if(crosshair == 1)
            rainbow_line_printf("CROSSHAIR: \033[1m\e[32mON\e[0m\n");
        else
            rainbow_line_printf("CROSSHAIR: \033[1m\e[31mOFF\e[0m\n");

        printf("\n");
    }
}

/***************************************************
   ~~ Program Entry Point
*/
int main(int argc, char *argv[])
{
    srand(time(0));

    // intro
    reprint();

    // open display 0
    d = XOpenDisplay(":0");
    if(d == NULL)
    {
        printf("Failed to open display\n");
        return 0;
    }

    // get default screen
    si = XDefaultScreen(d);

    // get graphics context
    gc = DefaultGC(d, si);

    // find window
    twin = findWindow(d, 0, "Counter-Strike");
    if(twin != 0)
        reprint();

    //
    
    XEvent event;
    memset(&event, 0x00, sizeof(event));

    //
    
    uint tc = 0;
    while(1)
    {
        // loop every SCAN_DELAY ms (1,000 microsecond = 1 millisecond)
        usleep(SCAN_DELAY);

        // bot toggle
        if(key_is_pressed(XK_Control_L) && key_is_pressed(XK_Alt_L))
        {
            if(enable == 0)
            {
                 // get window
                twin = findWindow(d, 0, "Counter-Strike");
                if(twin == 0)
                {
                    printf("Failed to detect a CS:GO window.\n");
                    sleep(1);
                    continue;
                }

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
                usleep(100000);
                reprint();
                speakS("on");
            }
            else
            {
                enable = 0;
                usleep(100000);
                reprint();
                speakS("off");
            }
        }
        
        // toggle bot on/off
        if(enable == 1)
        {
            // print samples per second when pressed
            if(key_is_pressed(XK_H))
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
                continue;
            }

            // input toggle
            if(key_is_pressed(XK_Control_R) && key_is_pressed(XK_Alt_R))
            {
                if(hotkeys == 0)
                {
                    hotkeys = 1;
                    usleep(100000);
                    reprint();
                    speakS("hk on");
                }
                else
                {
                    hotkeys = 0;
                    usleep(100000);
                    reprint();
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
                        usleep(100000);
                        reprint();
                        speakS("cx on");
                    }
                    else
                    {
                        crosshair = 0;
                        usleep(100000);
                        reprint();
                        speakS("cx off");
                    }
                }

                // sample capture toggle
                if(key_is_pressed(XK_L))
                {
                    if(sample_capture == 0)
                    {
                        char* home = getenv("HOME");
                        sprintf(targets_dir, "%s/Desktop/targets", home);
                        mkdir(targets_dir, 0777);
                        sample_capture = 1;
                        usleep(100000);
                        reprint();
                        speakS("sc on");
                    }
                    else
                    {
                        sample_capture = 0;
                        usleep(100000);
                        reprint();
                        speakS("sc off");
                    }
                }

                // sample capture
                static uint64_t scd = 0;
                if(sample_capture == 1 && key_is_pressed(XK_E) && microtime() > scd)
                {
                    char name[32];
                    sprintf(name, "%s/%i.ppm", targets_dir, rand());
#if PRELOG_SAVESAMPLE == 1
                    writePPM(name, &rgbbytes[0]);
#else
                    saveSample(name);
#endif
                    printf("\e[93mMANUAL SAVE:\e[38;5;123m %s\n", name);
                    scd = microtime() + 350000;
                }
            }
            
            if(hotkeys == 1 && key_is_pressed(XK_G)) // print activation when pressed
            {
                processScanArea(twin);
                const float ret = processModel(&input[0]);
                if(ret > ACTIVATION_SENITIVITY)
                {
                    printf("\e[93mA: %f\e[0m\n", ret);
                    XSetForeground(d, gc, 65280);
                    XDrawRectangle(d, event.xbutton.window, gc, x-rd2-1, y-rd2-1, r0+2, r0+2);
                    XSetForeground(d, gc, 0);
                    XDrawRectangle(d, event.xbutton.window, gc, x-rd2-2, y-rd2-2, r0+4, r0+4);
                    XFlush(d);
                }
                else
                {
                    printf("\e[0mA: %f\n", ret);
                    XSetForeground(d, gc, 16711680);
                    XDrawRectangle(d, event.xbutton.window, gc, x-rd2-1, y-rd2-1, r0+2, r0+2);
                    XSetForeground(d, gc, 0);
                    XDrawRectangle(d, event.xbutton.window, gc, x-rd2-2, y-rd2-2, r0+4, r0+4);
                    XFlush(d);
                }
            }
            else
            {
                if(crosshair == 1)
                {
                    XSetForeground(d, gc, 65280);
                    XDrawRectangle(d, event.xbutton.window, gc, x-rd2-1, y-rd2-1, r0+2, r0+2);
                    XSetForeground(d, gc, 0);
                    XDrawRectangle(d, event.xbutton.window, gc, x-rd2-2, y-rd2-2, r0+4, r0+4);
                    XFlush(d);
                }

                if(key_is_pressed(XK_W) == 0 && key_is_pressed(XK_A) == 0 && key_is_pressed(XK_S) == 0 && key_is_pressed(XK_D) == 0 && key_is_pressed(XK_Shift_L) == 0)
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
                            if(sample_capture == 1)
                            {
                                char name[32];
                                sprintf(name, "%s/%i.ppm", targets_dir, rand());
#if PRELOG_SAVESAMPLE == 1
                                writePPM(name, &rgbbytes[0]);
#else
                                saveSample(name);
#endif
                                printf("SAVED: %s\n", name);
                            }

                            // fire off as many shots as we need to
                            for(int i = 0; i < TRIGGER; i++)
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
                                usleep(TRIGGER_RATE_LIMIT_MS * 1000);
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
            }

        ///
        }
    }

    // done, never gets here in regular execution flow
    return 0;
}
