#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include "gc/gazelle_circuits.h"
#include "gc/gc.h"

#include <cryptoTools/Common/Defines.h>
#include <cryptoTools/Common/Log.h>
#include <cryptoTools/Common/Timer.h>

#include <cryptoTools/Network/Channel.h>
#include <cryptoTools/Network/IOService.h>
#include <cryptoTools/Network/Session.h>

#include <ot/cot_recv.h>
#include <ot/cot_send.h>
#include <ot/sr_base_ot.h>

using namespace osuCrypto;
using namespace lbcrypto;

std::string addr = "localhost";
u64 n_circ = 2304;
u64 layer_type = 1;

void buildGCLayer(GarbledCircuit& gc,
                  BuildContext& context,
                  ui64 width,
                  ui64 n_circ,
                  ui64 p) {
    switch (layer_type) {
        case 0:
            buildRELULayer(gc, context, width, n_circ, p);
            break;
        case 1:
            buildPool2Layer(gc, context, width, n_circ, p);
            break;
        case 2:
            buildSigmoidLayer(gc, context, width, n_circ, p);
            break;
        case 3:
            buildTanhLayer(gc, context, width, n_circ, p);
            break;
        default:
            std::cout << "Unknonw layer type" << std::endl;
            break;
    }
}

void gc_sender() {
    PRNG prng(_mm_set_epi32(4253465, 3434565, 234435, 23987045));
    setThreadName("Sender");

    // get up the networking
    IOService ios(0);
    Session sess(ios, addr, 1212, EpMode::Client);
    Channel chl = sess.addChannel();

    Timer time;
    chl.resetStats();
    time.setTimePoint("start");

    std::vector<block> baseRecv(128);
    BitVector baseChoice(128);
    baseChoice.randomize(prng);
    SRBaseOT base_ot;
    base_ot.receive(baseChoice, baseRecv, prng, chl);

    IKNPSender s;
    s.setBaseOts(baseRecv, baseChoice);

    // read empty circuit from file
    GarbledCircuit gc;
    BuildContext context;
    buildGCLayer(gc, context, 22, n_circ, 307201);

    // garble circuit
    InputLabels inputLabels(gc.n);
    BitVector outputBitMap(gc.m);
    garbleCircuit(&gc, inputLabels, outputBitMap);

    /*for(int i=0; i<gc.r; i++){
        std::cout << "gi " << i << " " << gc.wires[i].label0 << " " <<
    gc.wires[i].label1 <<std::endl;
    }*/

    std::cout << "      Sent: " << chl.getTotalDataSent() << std::endl
              << "  received: " << chl.getTotalDataRecv() << std::endl
              << std::endl;
    chl.resetStats();

    time.setTimePoint("setup");

    // transfer tables and labels
    chl.send(outputBitMap);
    // std::cout << "s out: " << outputBitMap << std::endl;

    std::vector<block> gc_constants = {gc.table_key, gc.wires[gc.n].label0,
                                       gc.wires[gc.n + 1].label1};
    chl.send(gc_constants);
    chl.send(gc.garbledTable);

    std::cout << "      Sent: " << chl.getTotalDataSent() << std::endl
              << "  received: " << chl.getTotalDataRecv() << std::endl
              << std::endl;
    chl.resetStats();
    time.setTimePoint("garble");

    // run ot
    span<std::array<block, 2>> in_c(inputLabels.data(), gc.n_c);
    s.send(in_c, prng, chl);

    BitVector in_s_choice(gc.n - gc.n_c);
    in_s_choice.randomize(prng);
    std::vector<block> in_s(gc.n - gc.n_c);
    for (ui64 i = 0; i < in_s.size(); i++) {
        in_s[i] = (in_s_choice[i]) ? inputLabels[gc.n_c + i][1]
                                   : inputLabels[gc.n_c + i][0];
    }
    chl.asyncSend(std::move(in_s));

    std::cout << "      Sent: " << chl.getTotalDataSent() << std::endl
              << "  received: " << chl.getTotalDataRecv() << std::endl
              << std::endl;
    chl.resetStats();
    time.setTimePoint("ot+eval");

    std::cout << time << std::endl;
    /*BitVector input_bits(gc.n);
    chl.recv(input_bits);
    // std::cout << "s in: " << input_bits << std::endl;
    std::vector<block> selectedLabels(gc.n);
    for(int i=0; i<gc.n; i++){
        selectedLabels[i] = inputLabels[i][input_bits[i]];
    }
    chl.send(selectedLabels);*/

    chl.close();
    sess.stop();
    ios.stop();
    return;
}

void gc_receiver() {
    PRNG prng(_mm_set_epi32(4253465, 3434565, 234435, 23987044));
    setThreadName("Receiver");

    // get up the networking
    IOService ios(0);
    Session sess(ios, addr, 1212, EpMode::Server);
    Channel chl = sess.addChannel();

    Timer time;
    chl.resetStats();
    time.setTimePoint("start");

    std::vector<std::array<block, 2>> baseSend(128);
    SRBaseOT send;
    send.send(baseSend, prng, chl);

    IKNPReceiver r;
    r.setBaseOts(baseSend);

    // read empty circuit from file
    GarbledCircuit gc;
    BuildContext context;
    buildGCLayer(gc, context, 22, n_circ, 307201);

    // get garbled tables and output maps
    BitVector outputBitMap(gc.m);
    chl.recv(outputBitMap);

    std::vector<block> gc_constants(3);
    chl.recv(gc_constants);
    gc.table_key = gc_constants[0];
    gc.wires[gc.n].label = gc_constants[1];
    gc.wires[gc.n + 1].label = gc_constants[2];

    chl.recv(gc.garbledTable);

    std::cout << "      Sent: " << chl.getTotalDataSent() << std::endl
              << "  received: " << chl.getTotalDataRecv() << std::endl
              << std::endl;
    chl.resetStats();

    time.setTimePoint("garbling");
    // pick inputs
    BitVector input_bits(gc.n_c);
    ExtractedLabels extractedLabels(gc.n);
    input_bits.randomize(prng);

    // run ot to get your labels
    span<block> in_c(extractedLabels.data(), gc.n_c);
    r.receive(input_bits, in_c, prng, chl);
    span<block> in_s(&extractedLabels[gc.n_c], gc.n - gc.n_c);
    chl.recv(in_s);
    /*chl.send(input_bits);
    chl.recv(extractedLabels);*/

    std::cout << "      Sent: " << chl.getTotalDataSent() << std::endl
              << "  received: " << chl.getTotalDataRecv() << std::endl
              << std::endl;
    chl.resetStats();

    time.setTimePoint("ot");
    // evaluate garbled circuit
    OutputLabels eval_outputs(gc.m);
    evaluate(&gc, extractedLabels, eval_outputs);

    // map outputs
    BitVector extractedMap(gc.m);
    for (int i = 0; i < gc.m; i++) {
        extractedMap[i] = outputBitMap[i] ^ getLSB(eval_outputs[i]);
    }

    std::cout << "      Sent: " << chl.getTotalDataSent() << std::endl
              << "  received: " << chl.getTotalDataRecv() << std::endl
              << std::endl;
    chl.resetStats();

    time.setTimePoint("eval");
    /*for(int i=0; i<gc.r; i++){
        std::cout << "ei " << i << " " << gc.wires[i].label <<std::endl;
    }*/
    std::cout << time << std::endl;
    std::cout << gc.n << " " << gc.m << " " << gc.q << " " << gc.r << std::endl;
    std::cout << gc.garbledTable.size() << std::endl;
    // std::cout << input_bits << std::endl;
    // std::cout << extractedMap << std::endl;

    chl.close();
    sess.stop();
    ios.stop();
    return;
}

int main(int argc, char** argv) {
    std::cout << "n_circ=";
    std::cin >> n_circ;
    std::cout << "layter_type=";
    std::cin >> layer_type;
    switch (layer_type) {
        case 0:
            std::cout << "ReLU layer" << std::endl;
            break;
        case 1:
            std::cout << "MaxPool layer" << std::endl;
            break;
        case 2:
            std::cout << "Sigmoid layer" << std::endl;
            break;
        case 3:
            std::cout << "Tanh layer" << std::endl;
            break;
    }

    if (argc == 1) {
        std::vector<std::thread> thrds(2);
        thrds[0] = std::thread([]() { gc_sender(); });
        thrds[1] = std::thread([]() { gc_receiver(); });
        for (auto& thrd : thrds)
            thrd.join();
    } else if (argc == 2) {
        int role = atoi(argv[1]);  // 0: send, 1: recv
        role ? gc_receiver() : gc_sender();
    } else {
        std::cout << "this program takes a runtime argument.\n\n"
                  << "to run the AES GC, run\n\n"
                  << "    gc-online [0|1]\n\n"
                  << "the optional {0,1} argument specifies in which case the "
                     "program will\n"
                  << "run between two terminals, where each one was set to the "
                     "opposite value. e.g.\n\n"
                  << "    gc-online 0\n\n"
                  << "    gc-online 1\n\n"
                  << "These programs are fully networked and try to connect at "
                     "localhost:1212.\n"
                  << std::endl;
    }
}
