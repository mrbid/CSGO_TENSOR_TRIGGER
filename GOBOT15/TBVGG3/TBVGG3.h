/*
--------------------------------------------------
    James William Fletcher (github.com/mrbid)
        AUGUST 2022 - TBVGG3
--------------------------------------------------
    Tiny Binary VGG3
    https://github.com/tfnn
    
    Release Notes:

        Output is a linear layer with sigmoid optional by specifying
        `#define SIGMOID_OUTPUT`.

        You can select between NORMAL_GLOROT or UNIFORM_GLOROT
        weight initialisation by specifying `#define UNIFORM_GLOROT`
        for uniform, otherwise normal is used by default.

        Sigmoid output is better for normalised inputs and a linear
        output is better for unnormalised inputs.

        There are three supported sizes of this network, 8, 16, and 32.
        You can select between them by defining `#define ADA8`, ADA16
        or ADA32.

    Information:

        This is an adaption inspired by the VGG series of networks.

        This VGG network is designed for binary classification and is
        only three layers deep. It uses Global Average Pooling rather
        than a final fully connected layer, additionally the final
        result is again just an average of the GAP. Essentially making
        this network an FCN version of the VGG network.

        The VGG network was originally created by the Visual Geometry Group
        of Oxford University in the United Kingdom. It was first proposed
        by Karen Simonyan and Andrew Zisserman, the original paper is
        available here; https://arxiv.org/abs/1409.1556

            TBVGG3 (ADA16)
            :: ReLU + 0 Padding
            28x28 x16
            > maxpool
            14x14 x32
            > maxpool
            7x7 x64
            > GAP + Average

        I like to call the gradient the error at times.

    Configuration;

        No batching of the forward passes before backproping.

        XAVIER GLOROT normal distribution weight initialisation.
        I read some places online that uniform GLOROT works
        better in CNN's, the truth is they both have their
        score sheet of gains and losses. I find normal is a
        smoother descent but uniform can reach lower losses
        although this is completely subjective to my bias.
        
        Since the original VGG paper references GLOROT with
        normal distribution, this is what I chose as the defacto.

        expected input RGB 28x28 pixels;
        float input[3][28][28];

    Preferences;

        You can see that I do not make an active effort to avoid
        branching, when I consider the trade off, such as with the
        TBVGG3_CheckPadded() check, I think to myself do I memcpy()
        to a new buffer with padding or include the padding in the
        original buffer or use branches to check if entering a padded
        coordinate, I chose the latter. I would rather a few extra
        branches than to bloat memory in some scenarios, although
        you can also see in TBVGG3_2x2MaxPool() that I choose a
        negligibly higher use of memory to avoid ALU divisions.

        I didn't think it was a good idea to maxpool the last
        layer because there are no fully connected layers,
        since it's going straight into a GAP it will make
        negligible difference in the final average. Maxpooling
        before a fully connected layer makes sense to reduce the
        amount of parameters to a more important subset. But this
        is a binary decision network, so a fully connected layer
        wont have a profound impact, we just want to know if our
        relevant features / filters had been activated enough to
        signal YES, if not, it's a NO.

    Comments;

        When it came to the back propagation I just worked it out
        using the knowledge and intuition I had gained from implementing
        back propagation in fully connected neural networks which is a
        in my opinion easier to understand. That's to say I didn't read
        or check any existing documentation for implementing back prop
        in CNN's. To be honest, the problem is something you can just
        see in your minds eye when you think about it. You know that
        you have to push a gradient backward and that process is very
        much the same as in fully connected layers.
        
        When a ReLU output is fed into a regular sigmoid function the
        output of the ReLU will always be >0 and thus the output of the
        sigmoid will always be 0.5 - 1.0, and the derivative will start
        at 0.25 and then reduce to 0 as the sigmoid input approaches 1.
        As such I have provided a suggested modification to the sigmoid
        function `1-(1 / expf(x))` which will insure that the output ranges
        from 0 to 1 and that the derivative will output 0.25 with an input
        of 0.5.

    Network size:

        ADA8:  23.6  KiB (24,128  bytes)
        ADA16: 92.1  KiB (94,336  bytes)
        ADA32: 364.2 KiB (372,992 bytes)

*/

#ifndef TBVGG3_H
#define TBVGG3_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define uint unsigned int
#define sint int

#ifndef LEARNING_RATE
    #define LEARNING_RATE 0.001f
#endif

#ifndef GAIN
    #define GAIN 0.0065f
#endif

#if !defined(ADA8) && !defined(ADA16) && !defined(ADA32)
    #define ADA16
#endif

#if !defined(OPTIM_NAG) && !defined(OPTIM_SGD) && !defined(OPTIM_ADA)
    #define OPTIM_ADA
#endif

#if defined(OPTIM_NAG) && !defined(NAG_MOMENTUM)
    #define NAG_MOMENTUM 0.1f
#endif

#ifdef ADA8
    #define L1 8
    #define L2 16
    #define L3 32
    #define RL3F 0.03125f // reciprocal L3 as float
#endif

#ifdef ADA16
    #define L1 16
    #define L2 32
    #define L3 64
    #define RL3F 0.015625f // reciprocal L3 as float
#endif

#ifdef ADA32
    #define L1 32
    #define L2 64
    #define L3 128
    #define RL3F 0.0078125f // reciprocal L3 as float
#endif

/*
--------------------------------------
    structures
--------------------------------------
*/

// network struct
struct
{
    //filters:num, d, w
    float l1f[L1][3 ][9];
    float l2f[L2][L1][9];
    float l3f[L3][L2][9];

    // filter bias's
    float l1fb[L1][1];
    float l2fb[L2][1];
    float l3fb[L3][1];
}
typedef TBVGG3_Network;

#define TBVGG3_LEARNTYPE float
#define LEARN_MAX 1.f
#define LEARN_MIN 0.f
#define NO_LEARN -1.f

/*
--------------------------------------
    functions
--------------------------------------
*/

float TBVGG3_Process(TBVGG3_Network* net, const float input[3][28][28], const TBVGG3_LEARNTYPE learn);
void  TBVGG3_Reset(TBVGG3_Network* net, const uint seed);
int   TBVGG3_SaveNetwork(TBVGG3_Network* net, const char* file);
int   TBVGG3_LoadNetwork(TBVGG3_Network* net, const char* file);
void  TBVGG3_Debug(TBVGG3_Network* net);

/*
--------------------------------------
    the code ...
--------------------------------------
*/

void TBVGG3_Debug(TBVGG3_Network* net)
{
    float min=0.f, avg=0.f, max=0.f;
    float recip_num_weights = 1.f/(L1*3*9);
    for(uint i = 0; i < L1; i++)
    {
        for(uint j = 0; j < 3; j++)
        {
            for(uint k = 0; k < 9; k++)
            {
                const float w = net->l1f[i][j][k];
                if(w < min){min = w;}
                else if(w > max){max = w;}
                avg += w;
            }
        }
    }
    printf("0: %+.3f %+.3f %+.3f [%+.3f]\n", min, avg*recip_num_weights, max, avg);

    min=0.f, avg=0.f, max=0.f;
    recip_num_weights = 1.f/(L2*L1*9);
    for(uint i = 0; i < L2; i++)
    {
        for(uint j = 0; j < L1; j++)
        {
            for(uint k = 0; k < 9; k++)
            {
                const float w = net->l2f[i][j][k];
                if(w < min){min = w;}
                else if(w > max){max = w;}
                avg += w;
            }
        }
    }
    printf("1: %+.3f %+.3f %+.3f [%+.3f]\n", min, avg*recip_num_weights, max, avg);

    min=0.f, avg=0.f, max=0.f;
    recip_num_weights = 1.f/(L3*L2*9);
    for(uint i = 0; i < L3; i++)
    {
        for(uint j = 0; j < L2; j++)
        {
            for(uint k = 0; k < 9; k++)
            {
                const float w = net->l3f[i][j][k];
                if(w < min){min = w;}
                else if(w > max){max = w;}
                avg += w;
            }
        }
    }
    printf("2: %+.3f %+.3f %+.3f [%+.3f]\n", min, avg*recip_num_weights, max, avg);
}

static inline float TBVGG3_RELU(const float x)
{
    if(x < 0.f){return 0.f;}
    return x;
}

static inline float TBVGG3_RELU_D(const float x)
{
    if(x > 0.f){return 1.f;}
    return 0.f;
}

#ifdef SIGMOID_OUTPUT
static inline float TBVGG3_SIGMOID(const float x)
{
    return 1.f-(1.f / expf(x));
}

static inline float TBVGG3_SIGMOID_D(const float x)
{
    return x * (1.f - x);
}
#endif

static inline float TBVGG3_OPTIM(const float input, const float error, float* momentum)
{
#ifdef OPTIM_ADA
    const float err = error * input;
    momentum[0] += err * err;
    return (LEARNING_RATE / sqrtf(momentum[0] + 1e-7f)) * err;
#endif
#ifdef OPTIM_NAG
    const float v = NAG_MOMENTUM * momentum[0] + ( LEARNING_RATE * error * input );
    const float n = v + NAG_MOMENTUM * (v - momentum[0]);
    momentum[0] = v;
    return n;
#endif
#ifdef OPTIM_SGD
    return LEARNING_RATE * error * input;
#endif
}

#ifdef UNIFORM_GLOROT
float TBVGG3_RandomWeight() // Uniform
{
    static const float rmax = 1.f/(float)RAND_MAX;
    float pr = 0.f;
    while(pr == 0.f) //never return 0
    {
        const float rv2 = ( ( ((float)rand()) * rmax ) * 2.f ) - 1.f;
        pr = roundf(rv2 * 100.f) * 0.01f; // two decimals of precision
    }
    return pr;
}
#else
float TBVGG3_RandomWeight() // Box Muller Normal
{
    static const float rmax = 1.f/(float)RAND_MAX;
    float u = ( ((float)rand()) * rmax) * 2.f - 1.f;
    float v = ( ((float)rand()) * rmax) * 2.f - 1.f;
    float r = u * u + v * v;
    while(r == 0.f || r > 1.f)
    {
        u = ( ((float)rand()) * rmax) * 2.f - 1.f;
        v = ( ((float)rand()) * rmax) * 2.f - 1.f;
        r = u * u + v * v;
    }
    return u * sqrtf(-2.f * logf(r) / r);
}
#endif

void TBVGG3_Reset(TBVGG3_Network* net, const uint seed)
{
    if(net == NULL){return;}

    // seed random
    if(seed == 0)
        srand(time(0));
    else
        srand(seed);

    // Weight Init
#ifdef UNIFORM_GLOROT
    const float dividend = 6.0f; // uniform
#else
    const float dividend = 2.0f; // normal
#endif

    //l1f
    float d = sqrtf(dividend / (3+L1));
    for(uint i = 0; i < L1; i++)
        for(uint j = 0; j < 3; j++)
            for(uint k = 0; k < 9; k++)
                net->l1f[i][j][k] = TBVGG3_RandomWeight() * d;

    //l2f
    d = sqrtf(dividend / (L1+L2));
    for(uint i = 0; i < L2; i++)
        for(uint j = 0; j < L1; j++)
            for(uint k = 0; k < 9; k++)
                net->l2f[i][j][k] = TBVGG3_RandomWeight() * d;

    //l3f
    d = sqrtf(dividend / (L2+L3));
    for(uint i = 0; i < L3; i++)
        for(uint j = 0; j < L2; j++)
            for(uint k = 0; k < 9; k++)
                net->l3f[i][j][k] = TBVGG3_RandomWeight() * d;

    // reset bias
    memset(net->l1fb, 0, sizeof(net->l1fb));
    memset(net->l2fb, 0, sizeof(net->l2fb));
    memset(net->l3fb, 0, sizeof(net->l3fb));
}

int TBVGG3_SaveNetwork(TBVGG3_Network* net, const char* file)
{
    if(net == NULL){return -1;}

    FILE* f = fopen(file, "wb");
    if(f == NULL)
        return -1;

    if(fwrite(net, 1, sizeof(TBVGG3_Network), f) != sizeof(TBVGG3_Network))
    {
        fclose(f);
        return -2;
    }

    fclose(f);
    return 0;
}

int TBVGG3_LoadNetwork(TBVGG3_Network* net, const char* file)
{
    if(net == NULL){return -1;}

    FILE* f = fopen(file, "rb");
    if(f == NULL)
        return -1;

    if(fread(net, 1, sizeof(TBVGG3_Network), f) != sizeof(TBVGG3_Network))
    {
        fclose(f);
        return -2;
    }

    fclose(f);
    return 0;
}

void TBVGG3_2x2MaxPool(const uint depth, const uint wh, const float input[depth][wh][wh], float output[depth][wh/2][wh/2])
{
    // for every depth
    for(uint d = 0; d < depth; d++)
    {
        // output tracking, more memory for less alu division ops
        uint oi = 0, oj = 0;

        // for every 2x2 chunk of input
        for(uint i = 0; i < wh; i += 2, oi++)
        {
            for(uint j = 0; j < wh; j += 2, oj++)
            {
                // get max val
                float max = 0.f;
                if(input[d][i][j] > max)
                    max = input[d][i][j];
                if(input[d][i][j+1] > max)
                    max = input[d][i][j+1];
                if(input[d][i+1][j] > max)
                    max = input[d][i+1][j];
                if(input[d][i+1][j+1] > max)
                    max = input[d][i+1][j+1];

                // output max val
                output[d][oi][oj] = max;
            }
            oj = 0;
        }
    }
}

static inline uint TBVGG3_CheckPadded(const sint x, const sint y, const uint wh)
{
    if(x < 0 || y < 0 || x >= wh || y >= wh)
        return 1;
    return 0;
}

float TBVGG3_3x3Conv(const uint depth, const uint wh, const float input[depth][wh][wh], const uint y, const uint x, const float filter[depth][9], const float* filter_bias)
{
    // input depth needs to be same as filter depth
    // This will return a single float output. Call this x*y times per filter.
    // It's zero padded so if TBVGG3_CheckPadded() returns 1 it's a no operation
    float ro = 0.f;
    sint nx = 0, ny = 0;
    for(uint i = 0; i < depth; i++)
    {
        // lower row
        nx = x-1, ny = y-1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            ro += input[i][ny][nx] * filter[i][0];

        nx = x,   ny = y-1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            ro += input[i][ny][nx] * filter[i][1];

        nx = x+1, ny = y-1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            ro += input[i][ny][nx] * filter[i][2];

        // middle row
        nx = x-1, ny = y;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            ro += input[i][ny][nx] * filter[i][3];

        nx = x,   ny = y;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            ro += input[i][ny][nx] * filter[i][4];

        nx = x+1, ny = y;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            ro += input[i][ny][nx] * filter[i][5];

        // top row
        nx = x-1, ny = y+1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            ro += input[i][ny][nx] * filter[i][6];

        nx = x,   ny = y+1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            ro += input[i][ny][nx] * filter[i][7];

        nx = x+1, ny = y+1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            ro += input[i][ny][nx] * filter[i][8];
    }

    // bias
    ro += filter_bias[0];

    // return output
    return TBVGG3_RELU(ro);
}

void TBVGG3_3x3ConvB(const uint depth, const uint wh, const float input[depth][wh][wh], const float error[depth][wh][wh], const uint y, const uint x, float filter[depth][9], float filter_momentum[depth][9], float* bias, float* bias_momentum)
{
    // backprop version
    sint nx = 0, ny = 0;
    for(uint i = 0; i < depth; i++)
    {
        // lower row
        nx = x-1, ny = y-1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            filter[i][0] += TBVGG3_OPTIM(input[i][ny][nx], error[i][y][x], &filter_momentum[i][0]);
            
        nx = x,   ny = y-1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            filter[i][1] += TBVGG3_OPTIM(input[i][ny][nx], error[i][y][x], &filter_momentum[i][1]);

        nx = x+1, ny = y-1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            filter[i][2] += TBVGG3_OPTIM(input[i][ny][nx], error[i][y][x], &filter_momentum[i][2]);

        // middle row
        nx = x-1, ny = y;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            filter[i][3] += TBVGG3_OPTIM(input[i][ny][nx], error[i][y][x], &filter_momentum[i][3]);

        nx = x,   ny = y;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            filter[i][4] += TBVGG3_OPTIM(input[i][ny][nx], error[i][y][x], &filter_momentum[i][4]);

        nx = x+1, ny = y;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            filter[i][5] += TBVGG3_OPTIM(input[i][ny][nx], error[i][y][x], &filter_momentum[i][5]);

        // top row
        nx = x-1, ny = y+1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            filter[i][6] += TBVGG3_OPTIM(input[i][ny][nx], error[i][y][x], &filter_momentum[i][6]);

        nx = x,   ny = y+1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            filter[i][7] += TBVGG3_OPTIM(input[i][ny][nx], error[i][y][x], &filter_momentum[i][7]);

        nx = x+1, ny = y+1;
        if(TBVGG3_CheckPadded(nx, ny, wh) == 0)
            filter[i][8] += TBVGG3_OPTIM(input[i][ny][nx], error[i][y][x], &filter_momentum[i][8]);
        
        // bias
        bias[0] += TBVGG3_OPTIM(1, error[i][y][x], bias_momentum);
    }
}

float TBVGG3_Process(TBVGG3_Network* net, const float input[3][28][28], const TBVGG3_LEARNTYPE learn)
{
    if(net == NULL){return -1.337f;}

    // filter momentum's
    float l1fm[L1][3 ][9]={0};
    float l2fm[L2][L1][9]={0};
    float l3fm[L3][L2][9]={0};

    // filter bias momentum's
    float l1fbm[L1][1]={0};
    float l2fbm[L2][1]={0};
    float l3fbm[L3][1]={0};

    // outputs
    //       d,  y,  x
    float o1[L1][28][28];
        float p1[L1][14][14]; // pooled
    float o2[L2][14][14];
        float p2[L2][7][7];   // pooled
    float o3[L3][7][7];

    // error gradients
    //       d,  y,  x
    float e1[L1][28][28];
    float e2[L2][14][14];
    float e3[L3][7][7];

    // convolve input with L1 filters
    for(uint i = 0; i < L1; i++) // num filter
    {
        for(uint j = 0; j < 28; j++) // height
        {
            for(uint k = 0; k < 28; k++) // width
            {
                o1[i][j][k] = TBVGG3_3x3Conv(3, 28, input, j, k, net->l1f[i], net->l1fb[i]);
            }
        }
    }

    // max pool the output
    TBVGG3_2x2MaxPool(L1, 28, o1, p1);

    // convolve output with L2 filters
    for(uint i = 0; i < L2; i++) // num filter
    {
        for(uint j = 0; j < 14; j++) // height
        {
            for(uint k = 0; k < 14; k++) // width
            {
                o2[i][j][k] = TBVGG3_3x3Conv(L1, 14, p1, j, k, net->l2f[i], net->l2fb[i]);
            }
        }
    }

    // max pool the output
    TBVGG3_2x2MaxPool(L2, 14, o2, p2);

    // convolve output with L3 filters
    for(uint i = 0; i < L3; i++) // num filter
    {
        for(uint j = 0; j < 7; j++) // height
        {
            for(uint k = 0; k < 7; k++) // width
            {
                o3[i][j][k] = TBVGG3_3x3Conv(L2, 7, p2, j, k, net->l3f[i], net->l3fb[i]);
            }
        }
    }

    // global average pooling
    float gap[L3] = {0};
    for(uint i = 0; i < L3; i++)
    {
        for(uint j = 0; j < 7; j++)
            for(uint k = 0; k < 7; k++)
                gap[i] += o3[i][j][k];
        gap[i] *= 0.02040816285f; // 1/49
    }

    // average final activation
    float output = 0.f;
    for(uint i = 0; i < L3; i++)
        output += gap[i];
    output *= RL3F;

#ifdef SIGMOID_OUTPUT
    output = TBVGG3_SIGMOID(output);
#endif

    // return activation else backprop
    if(learn == NO_LEARN)
    {
        return output;
    }
    else
    {
        // error/gradient slope scaled by derivative
#ifdef SIGMOID_OUTPUT
        const float g0 = TBVGG3_SIGMOID_D(output) * (learn - output);
        //printf("g0: %f %f %f %f %f\n", g0, learn, output, (learn - output), TBVGG3_SIGMOID_D(output));
#else
        float g0 = learn - output;
        //printf("g0: %f %f %f %f\n", g0, learn, output, (learn - output));
#endif

        // ********** Gradient Back Pass **********

        // layer 3
        float l3er = 0.f;
        for(uint i = 0; i < L3; i++) // num filter
        {
            for(uint j = 0; j < 7; j++) // height
            {
                for(uint k = 0; k < 7; k++) // width
                {
                    // set error
                    e3[i][j][k] = GAIN * TBVGG3_RELU_D(o3[i][j][k]) * g0;

                    // every output error gradient for every filter weight :: per filter
                    for(uint d = 0; d < L2; d++) // depth
                        for(uint w = 0; w < 9; w++) // weight
                            l3er += net->l3f[i][d][w] * e3[i][j][k];
                    l3er += net->l3fb[i][0] * e3[i][j][k];
                }
            }
        }

        // layer 2
        float l2er = 0.f;
        for(uint i = 0; i < L2; i++) // num filter
        {
            for(uint j = 0; j < 14; j++) // height
            {
                for(uint k = 0; k < 14; k++) // width
                {
                    // set error
                    e2[i][j][k] = GAIN * TBVGG3_RELU_D(o2[i][j][k]) * l3er;

                    // every output error gradient for every filter weight :: per filter
                    for(uint d = 0; d < L1; d++) // depth
                        for(uint w = 0; w < 9; w++) // weight
                            l2er += net->l2f[i][d][w] * e2[i][j][k];
                    l2er += net->l2fb[i][0] * e2[i][j][k];
                }
            }
        }

        // layer 1
        for(uint i = 0; i < L1; i++) // num filter
        {
            for(uint j = 0; j < 28; j++) // height
                for(uint k = 0; k < 28; k++) // width
                    e1[i][j][k] = GAIN * TBVGG3_RELU_D(o1[i][j][k]) * l2er; // set error
        }

        // ********** Weight Nudge Forward Pass **********
        
        // convolve filter 1 with layer 1 error gradients
        for(uint i = 0; i < L1; i++) // num filter
        {
            for(uint j = 0; j < 28; j++) // height
                for(uint k = 0; k < 28; k++) // width
                    TBVGG3_3x3ConvB(3, 28, input, e1, j, k, net->l1f[i], l1fm[i], net->l1fb[i], l1fbm[i]);
        }

        // convolve filter 2 with layer 2 error gradients
        for(uint i = 0; i < L2; i++) // num filter
        {
            for(uint j = 0; j < 14; j++) // height
                for(uint k = 0; k < 14; k++) // width
                    TBVGG3_3x3ConvB(L1, 14, o1, e2, j, k, net->l2f[i], l2fm[i], net->l2fb[i], l2fbm[i]);
        }

        // convolve filter 3 with layer 3 error gradients
        for(uint i = 0; i < L3; i++) // num filter
        {
            for(uint j = 0; j < 7; j++) // height
                for(uint k = 0; k < 7; k++) // width
                    TBVGG3_3x3ConvB(L2, 7, o2, e3, j, k, net->l3f[i], l3fm[i], net->l3fb[i], l3fbm[i]);
        }
        
        // weights nudged
    }

    // return activation
    return output;
}

#endif
