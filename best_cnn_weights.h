#ifndef csgo_mrbid_net
#define csgo_mrbid_net

#include <stdio.h>

/*
    James William Fletcher
    github.com/mrbid - September 2022
    This is a forward pass network I have spent a lot of
    time working on from scratch. As of this month and
    year I consider this to be the best performing
    minimal network.

    no padding, no bias, grayscale input, 3x3 kernels, channels_last
    rows, cols, kernels
    y,x,k

    kernels:
    3,3,2
    3,3,4
    3,3,8

    input:   28x28x1
    conv:    3x3x2 > 26x26x2 : relu
    maxpool: 13x13
    conv:    3x3x4 > 11x11x4 : relu
    maxpool: 5x5
    conv:    3x3x8 > 3x3x8 : relu
    gap:     array(8)
    dense:   sigmoid output

*/

const float c1[] = {-0.10085268,-0.5327633,-0.093879305,0.9093193,0.45742872,-0.54601395,-0.109133825,0.27121156,-0.3521443,-0.44989288,0.37559956,0.28491852,0.8056581,0.36666352,0.5286602,-0.60915345,-0.11539043,0.22436628};
// 18

const float c2[] = {0.3625854,-0.2961693,0.34357676,-0.09004362,0.51307756,-3.5217664,-0.39044002,-0.2900187,0.13758732,-0.32059073,-0.10238134,0.37341678,1.2980468,-1.8488529,-1.5052714,0.7356388,-0.078636244,0.27923414,0.077999316,0.4445,-0.36241,0.9078533,-0.38832384,0.21853346,-0.46116298,0.30799797,0.5373813,0.37880236,-0.7928643,-0.8725977,0.2778252,0.097840056,0.42987445,0.14128259,0.36648455,-0.22519167,0.15652227,0.92756784,0.23862366,0.6194639,0.30409643,-0.51745474,0.0024263235,-0.12897015,1.1616807,0.7814885,-0.62208676,-0.40302494,-0.16045721,0.2694907,-0.21484402,0.11004151,0.764975,0.33624822,-0.19113706,1.9655595,-0.46846592,0.22514774,0.43947503,-1.0024278,0.36089367,0.15557878,-0.2792994,0.12491965,-0.22553426,-0.3300921,-0.017703544,-0.7742302,1.2367371,0.1950125,-1.3916389,-0.2607476};
// 72

const float c3[] = {-0.6666318,-0.6499509,0.017249504,-0.16326442,-0.22811371,0.6105372,-1.2061936,0.010394949,1.9372067,1.7710887,1.7254974,1.7425989,-1.3244041,1.048707,2.1588702,1.7392019,-0.42127144,-0.11061933,-0.3554936,-0.019667566,0.12338132,-0.06739373,0.20636322,-0.05645989,0.90321827,1.4634635,1.3782281,1.2930415,-1.1278466,0.7545569,0.82316697,1.4057527,0.38134363,-0.64948016,0.42578185,-0.23418133,0.30430847,-0.31941235,0.017438103,-0.33156964,0.5274514,1.8053904,0.786523,1.5999506,-1.629409,1.9005216,1.5866919,1.8491492,0.45213234,0.15404697,0.5348579,-0.03504686,0.50366986,-0.38578567,0.09311065,0.14972478,0.06680024,0.24241066,-0.5661344,-0.46240142,0.6674583,-0.061471906,0.7778293,0.00547942,0.3505525,-0.3217814,0.5647048,-0.46430093,0.3413247,-0.93530357,0.68285865,-0.8179386,-0.9443577,0.99222124,-0.74951744,1.5060915,-1.8539797,-1.2733712,-0.3136488,1.6198969,0.3050006,-0.12353183,0.2600941,-0.05577971,0.7012368,-0.69120103,0.08819378,-0.23662797,-0.9264782,0.02257323,-1.304195,-0.266403,0.35280138,-0.93557376,-0.2724371,0.036498073,-1.0345271,-0.09567977,-0.6391683,-0.14270216,-0.11629022,0.13955416,-0.72821933,-0.17132543,0.35356653,-0.2813003,-0.60093063,-0.18304347,0.14200652,1.4064893,0.32346547,-0.5312404,-0.23669605,-0.11699825,-0.29569745,-0.25634912,0.30916855,0.4422385,-0.106391065,-0.16948828,0.06795499,0.5405564,-0.07677913,0.23410861,-0.30062374,1.1781301,-1.1384887,-0.017022943,-0.30742258,-0.17424811,-0.011861075,-0.006211239,0.07467434,-0.4911204,-0.42237592,0.19911426,0.94744396,0.56354165,0.12879127,0.69064784,-0.7407655,1.937518,0.3084967,0.844227,-0.121149786,0.2743653,0.38621083,0.3470594,0.056427915,-0.12770227,-0.13791257,0.23218161,0.7226025,-0.22627075,-0.246187,-0.7509091,0.97558045,0.75327444,0.8887772,-0.7208585,0.9673093,-0.2824686,0.158171,-0.63391435,0.56319493,0.23208329,0.48720482,-0.33179942,-0.34003234,0.17145425,0.107116565,0.5767645,0.17950015,-0.58480746,-0.886015,0.3794311,0.037069265,-0.03622643,0.08028794,0.24633844,0.10488024,0.17855826,-0.113084584,0.13750535,-0.23423539,-0.012242726,-0.26542222,-0.1018701,0.8441302,0.29523608,-0.32725728,0.040206067,0.0357823,0.0075954227,-0.23309204,0.24467848,-1.1471972,-0.1885725,-0.17840974,0.31720218,1.2253202,-0.33104882,0.3171273,-0.25064176,-0.41660643,1.2757696,1.1084087,-0.033284497,0.18243246,-0.09396731,-0.09534969,-0.1469201,0.33480045,0.6946161,0.2978034,-0.36039257,1.0407037,0.1910912,1.3607627,0.48659712,-2.0204294,0.6403378,-0.08928161,0.9058087,-0.24339463,0.5362985,-0.0608694,0.33523193,-1.1249162,-1.2907683,0.14169537,0.12229522,0.10730731,0.5749354,0.36680436,0.5716122,-0.9392891,1.2686868,0.9139534,0.20032431,-0.58617896,0.2567366,-0.50432104,0.10609374,0.32372996,-0.20171365,-0.33536744,0.31133586,0.61647445,-0.33804142,0.35246393,-0.615248,0.0524327,0.66400695,1.0802368,-0.8643,0.43973807,0.5051888,0.1118751,0.49338824,-0.8811681,-2.5034041,0.49740288,0.46392873,-0.4952641,-0.123362236,-0.8946474,-0.195359,0.07260272,-1.8585433,-1.1545897,0.36304718,-0.7406244,0.07727978,-0.48872453,-0.045256052,0.4778138,-0.15295202,-0.7671279,0.13332541,-1.1735713,-0.22989619,-0.6292691,-0.5701288,0.37870747,-0.35531673,0.4285855,-0.5731319};
// 288

const float d1[] = {2.3085554,1.15268,2.1598685,1.219241,-1.2499405,-3.7682545,2.529647,1.0187337};
// 8

#endif
