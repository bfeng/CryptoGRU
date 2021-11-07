#include "grucell.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <random>
#include <string>
#include "gc/gazelle_circuits.h"
#include "gc/gc.h"
#include "math/automorph.h"
#include "math/bit_twiddle.h"
#include "math/distrgen.h"
#include "math/params.h"
#include "math/transfrm.h"
#include "pke/encoding.h"
#include "utils/debug.h"
#include "utils/test.h"

#include "pke/conv1d.h"
#include "pke/conv2d.h"
#include "pke/encoding.h"
#include "pke/fv.h"
#include "pke/gemm.h"
#include "pke/layers.h"
#include "pke/mat_mul.h"
#include "pke/pke_types.h"
#include "pke/square.h"

#include <cryptoTools/Common/Log.h>
#include <cryptoTools/Common/Timer.h>
#include <cryptoTools/Network/Channel.h>
#include <cryptoTools/Network/IOService.h>
#include <cryptoTools/Network/Session.h>

#include <ot/cot_recv.h>
#include <ot/cot_send.h>
#include <ot/sr_base_ot.h>

namespace lbcrypto {

using namespace std;
using namespace osuCrypto;

string addr = "localhost";
// ui32 mat_window_size = 10, mat_num_windows = 2;
ui32 num_rep = 100;

void ahe_client(int port, ui32 num_rows, ui32 num_cols, ui32 window_size) {
    cout << "Client" << endl;

    DiscreteGaussianGenerator dgg = DiscreteGaussianGenerator(4.0);
    FVParams test_params{
        true,       opt::q,
        opt::p,     opt::logn,
        opt::phim,  (opt::q / opt::p),
        OPTIMIZED,  make_shared<DiscreteGaussianGenerator>(dgg),
        window_size};
    ui32 num_windows = 1 + floor(log2(test_params.q)) / test_params.window_size;

    // get up the networking
    IOService ios(0);
    Session sess(ios, addr, port, EpMode::Client);
    Channel chl = sess.addChannel();

    uv64 vec = get_dgg_testvector(num_cols, opt::p);

    Timer time;
    chl.resetStats();
    time.setTimePoint("start");
    // KeyGen
    auto kp = KeyGen(test_params);

    ui32 num_rot = nxt_pow2(num_rows) * nxt_pow2(num_cols) / opt::phim;
    uv32 index_list;
    for (ui32 i = 1; i < num_rot; i++) {
        index_list.push_back(i);
    }
    for (ui32 i = num_rot; i < num_cols; i *= 2) {
        index_list.push_back(i);
    }

    EvalAutomorphismKeyGen(kp.sk, index_list, test_params);
    for (ui32 n = 0; n < index_list.size(); n++) {
        auto rk = g_rk_map[index_list[n]];
        for (ui32 w = 0; w < num_windows; w++) {
            chl.send(rk->a[w]);
            chl.send(rk->b[w]);
        }
    }

    cout << "      Sent: " << chl.getTotalDataSent() << endl
         << "  received: " << chl.getTotalDataRecv() << endl
         << endl;
    chl.resetStats();

    time.setTimePoint("setup");

    for (ui32 rep = 0; rep < num_rep; rep++) {
        auto ct_vec =
            preprocess_vec(kp.sk, vec, window_size, num_windows, test_params);
        for (ui32 n = 0; n < ct_vec.size(); n++) {
            chl.send(ct_vec[n].a);
            chl.send(ct_vec[n].b);
        }

        Ciphertext ct_prod(opt::phim);
        chl.recv(ct_prod.a);
        chl.recv(ct_prod.b);
        auto prod =
            postprocess_prod(kp.sk, ct_prod, num_cols, num_rows, test_params);
    }

    cout << "      Sent: " << chl.getTotalDataSent() << endl
         << "  received: " << chl.getTotalDataRecv() << endl
         << endl;
    chl.resetStats();

    time.setTimePoint("online");

    cout << time << endl;

    chl.close();
    sess.stop();
    ios.stop();
    return;
}

void ahe_server(int port, ui32 num_rows, ui32 num_cols, ui32 window_size) {
    cout << "Server" << endl;

    DiscreteGaussianGenerator dgg = DiscreteGaussianGenerator(4.0);
    FVParams test_params{
        true,       opt::q,
        opt::p,     opt::logn,
        opt::phim,  (opt::q / opt::p),
        OPTIMIZED,  make_shared<DiscreteGaussianGenerator>(dgg),
        window_size};
    ui32 num_windows = 1 + floor(log2(test_params.q)) / test_params.window_size;

    // get up the networking
    IOService ios(0);
    Session sess(ios, addr, port, EpMode::Server);
    Channel chl = sess.addChannel();

    Timer time;
    time.setTimePoint("start");

    vector<uv64> mat(num_rows, uv64(num_cols));
    for (ui32 row = 0; row < num_rows; row++) {
        mat[row] = get_dgg_testvector(num_cols, opt::p);
    }
    auto enc_mat =
        preprocess_matrix(mat, window_size, num_windows, test_params);

    ui32 num_rot = nxt_pow2(num_rows) * nxt_pow2(num_cols) / opt::phim;
    uv32 index_list;
    for (ui32 i = 1; i < num_rot; i++) {
        index_list.push_back(i);
    }
    for (ui32 i = num_rot; i < num_cols; i *= 2) {
        index_list.push_back(i);
    }

    for (ui32 n = 0; n < index_list.size(); n++) {
        RelinKey rk(test_params.phim, num_windows);
        for (ui32 w = 0; w < num_windows; w++) {
            chl.recv(rk.a[w]);
            chl.recv(rk.b[w]);
        }
        g_rk_map[index_list[n]] = make_shared<RelinKey>(rk);
    }

    time.setTimePoint("setup");
    for (ui32 rep = 0; rep < num_rep; rep++) {
        CTVec ct_vec(num_windows, Ciphertext(opt::phim));
        for (ui32 n = 0; n < ct_vec.size(); n++) {
            chl.recv(ct_vec[n].a);
            chl.recv(ct_vec[n].b);
        }

        auto ct_prod = mat_mul_online(ct_vec, enc_mat, num_cols, test_params);
        chl.send(ct_prod.a);
        chl.send(ct_prod.b);
    }
    time.setTimePoint("online");

    cout << time << endl;

    chl.close();
    sess.stop();
    ios.stop();
    return;
}

void gc_sender(int port, int layer_type, u64 n_circ) {
    PRNG prng(_mm_set_epi32(4253465, 3434565, 234435, 23987045));
    setThreadName("Sender");

    // get up the networking
    IOService ios(0);
    Session sess(ios, addr, port, EpMode::Client);
    Channel chl = sess.addChannel();

    Timer time;
    chl.resetStats();
    time.setTimePoint("start");

    vector<block> baseRecv(128);
    BitVector baseChoice(128);
    baseChoice.randomize(prng);
    SRBaseOT base_ot;
    base_ot.receive(baseChoice, baseRecv, prng, chl);

    IKNPSender s;
    s.setBaseOts(baseRecv, baseChoice);

    // read empty circuit from file
    GarbledCircuit gc;
    BuildContext context;
    switch (layer_type) {
        case 1:
            buildPool2Layer(gc, context, 22, n_circ, 307201);
            break;
        case 2:
            buildRELULayer(gc, context, 22, n_circ, 307201);
            break;
        case 3:
            buildSigmoidLayer(gc, context, 22, n_circ, 307201);
            break;
        case 4:
            buildTanhLayer(gc, context, 22, n_circ, 307201);
            break;
    }

    // garble circuit
    InputLabels inputLabels(gc.n);
    BitVector outputBitMap(gc.m);
    garbleCircuit(&gc, inputLabels, outputBitMap);

    cout << "      Sent: " << chl.getTotalDataSent() << endl
         << "  received: " << chl.getTotalDataRecv() << endl
         << endl;
    chl.resetStats();

    time.setTimePoint("setup");

    // transfer tables and labels
    chl.send(outputBitMap);
    // cout << "s out: " << outputBitMap << endl;

    vector<block> gc_constants = {gc.table_key, gc.wires[gc.n].label0,
                                  gc.wires[gc.n + 1].label1};
    chl.send(gc_constants);
    chl.send(gc.garbledTable);

    cout << "      Sent: " << chl.getTotalDataSent() << endl
         << "  received: " << chl.getTotalDataRecv() << endl
         << endl;
    chl.resetStats();
    time.setTimePoint("garble");

    // run ot
    span<array<block, 2>> in_c(inputLabels.data(), gc.n_c);
    s.send(in_c, prng, chl);

    BitVector in_s_choice(gc.n - gc.n_c);
    in_s_choice.randomize(prng);
    vector<block> in_s(gc.n - gc.n_c);
    for (ui64 i = 0; i < in_s.size(); i++) {
        in_s[i] = (in_s_choice[i]) ? inputLabels[gc.n_c + i][1]
                                   : inputLabels[gc.n_c + i][0];
    }
    chl.asyncSend(move(in_s));

    cout << "      Sent: " << chl.getTotalDataSent() << endl
         << "  received: " << chl.getTotalDataRecv() << endl
         << endl;
    chl.resetStats();
    time.setTimePoint("ot+eval");

    cout << time << endl;

    chl.close();
    sess.stop();
    ios.stop();
    return;
}

void gc_receiver(int port, int layer_type, u64 n_circ) {
    PRNG prng(_mm_set_epi32(4253465, 3434565, 234435, 23987044));
    setThreadName("Receiver");

    // get up the networking
    IOService ios(0);
    Session sess(ios, addr, port, EpMode::Server);
    Channel chl = sess.addChannel();

    Timer time;
    chl.resetStats();
    time.setTimePoint("start");

    vector<array<block, 2>> baseSend(128);
    SRBaseOT send;
    send.send(baseSend, prng, chl);

    IKNPReceiver r;
    r.setBaseOts(baseSend);

    // read empty circuit from file
    GarbledCircuit gc;
    BuildContext context;
    switch (layer_type) {
        case 1:
            buildPool2Layer(gc, context, 22, n_circ, 307201);
            break;
        case 2:
            buildRELULayer(gc, context, 22, n_circ, 307201);
            break;
        case 3:
            buildSigmoidLayer(gc, context, 22, n_circ, 307201);
            break;
        case 4:
            buildTanhLayer(gc, context, 22, n_circ, 307201);
            break;
    }

    // get garbled tables and output maps
    BitVector outputBitMap(gc.m);
    chl.recv(outputBitMap);

    vector<block> gc_constants(3);
    chl.recv(gc_constants);
    gc.table_key = gc_constants[0];
    gc.wires[gc.n].label = gc_constants[1];
    gc.wires[gc.n + 1].label = gc_constants[2];

    chl.recv(gc.garbledTable);

    cout << "      Sent: " << chl.getTotalDataSent() << endl
         << "  received: " << chl.getTotalDataRecv() << endl
         << endl;
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

    cout << "      Sent: " << chl.getTotalDataSent() << endl
         << "  received: " << chl.getTotalDataRecv() << endl
         << endl;
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

    cout << "      Sent: " << chl.getTotalDataSent() << endl
         << "  received: " << chl.getTotalDataRecv() << endl
         << endl;
    chl.resetStats();

    time.setTimePoint("eval");
    cout << time << endl;
    cout << gc.n << " " << gc.m << " " << gc.q << " " << gc.r << endl;
    cout << gc.garbledTable.size() << endl;

    chl.close();
    sess.stop();
    ios.stop();
    return;
}

void GRUCell::matmul(ui32 num_rows, ui32 num_cols, ui32 window_size) {
    ftt_precompute(opt::z, opt::q, opt::logn);
    ftt_precompute(opt::z_p, opt::p, opt::logn);
    encoding_precompute(opt::p, opt::logn);
    precompute_automorph_index(opt::phim);

    vector<thread> thrds(2);
    thrds[0] = thread([num_rows, num_cols, window_size]() {
        ahe_server(1211, num_rows, num_cols, window_size);
    });
    thrds[1] = thread([num_rows, num_cols, window_size]() {
        ahe_client(1211, num_rows, num_cols, window_size);
    });

    for (auto& thrd : thrds)
        thrd.join();
}

void GRUCell::elem_add(ui32 vec_size) {
    cout << "NN Layers Benchmark (ms):" << endl;

    //------------------ Setup Parameters ------------------
    ui64 nRep = 1;
    double start, stop;

    ui64 z = RootOfUnity(opt::phim << 1, opt::q);
    ui64 z_p = RootOfUnity(opt::phim << 1, opt::p);
    ftt_precompute(z, opt::q, opt::logn);
    ftt_precompute(z_p, opt::p, opt::logn);
    encoding_precompute(opt::p, opt::logn);
    precompute_automorph_index(opt::phim);

    DiscreteGaussianGenerator dgg = DiscreteGaussianGenerator(4.0);

    FVParams slow_params{false,     opt::q,
                         opt::p,    opt::logn,
                         opt::phim, (opt::q / opt::p),
                         OPTIMIZED, make_shared<DiscreteGaussianGenerator>(dgg),
                         8};

    FVParams fast_params = slow_params;
    fast_params.fast_modulli = true;

    FVParams test_params = fast_params;

    //------------------- Synthetic Data -------------------
    uv64 vec_c = get_dgg_testvector(vec_size, opt::p);
    uv64 vec_s = get_dgg_testvector(vec_size, opt::p);

    //----------------------- KeyGen -----------------------
    nRep = 10;
    auto kp = KeyGen(test_params);

    start = currentDateTime();
    for (ui64 i = 0; i < nRep; i++) {
        kp = KeyGen(test_params);
    }
    stop = currentDateTime();
    cout << " KeyGen: " << (stop - start) / nRep << endl;

    //----------------- Client Preprocess ------------------
    nRep = 100;
    auto ct_vec = preprocess_client_share(kp.sk, vec_c, test_params);
    start = currentDateTime();
    for (ui64 i = 0; i < nRep; i++) {
        ct_vec = preprocess_client_share(kp.sk, vec_c, test_params);
    }
    stop = currentDateTime();
    cout << " Preprocess Client: " << (stop - start) / nRep << endl;

    //----------------- Server Preprocess -----------------
    vector<uv64> pt_vec;
    uv64 vec_s_f;
    tie(pt_vec, vec_s_f) = preprocess_server_share(vec_s, test_params);
    start = currentDateTime();
    for (ui64 i = 0; i < nRep; i++) {
        tie(pt_vec, vec_s_f) = preprocess_server_share(vec_s, test_params);
    }
    stop = currentDateTime();
    cout << " Preprocess Server: " << (stop - start) / nRep << endl;

    //---------------------- Square -----------------------
    auto ct_c_f = square_online(ct_vec, pt_vec, test_params);
    start = currentDateTime();
    for (ui64 i = 0; i < nRep; i++) {
        ct_c_f = square_online(ct_vec, pt_vec, test_params);
    }
    stop = currentDateTime();
    cout << " Multiply: " << (stop - start) / nRep << endl;

    //------------------- Post-Process ---------------------
    auto vec_c_f =
        postprocess_client_share(kp.sk, ct_c_f, vec_size, test_params);
    start = currentDateTime();
    for (ui64 i = 0; i < nRep; i++) {
        vec_c_f =
            postprocess_client_share(kp.sk, ct_c_f, vec_size, test_params);
    }
    stop = currentDateTime();
    cout << " Post-Process: " << (stop - start) / nRep << endl;

    //--------------------- Square PT ----------------------
    start = currentDateTime();
    auto vec_c_f_ref = square_pt(vec_c, vec_s, vec_s_f, opt::p);
    for (ui64 i = 0; i < nRep; i++) {
        vec_c_f_ref = square_pt(vec_c, vec_s, vec_s_f, opt::p);
    }
    stop = currentDateTime();
    cout << " Multiply PT: " << (stop - start) / nRep << endl;

    //----------------------- Check ------------------------
    cout << endl;

    check_vec_eq(vec_c_f_ref, vec_c_f, "square mismatch:\n");
}

void GRUCell::elem_mult(ui32 vec_size) {
    cout << "NN Layers Benchmark (ms):" << endl;

    //------------------ Setup Parameters ------------------
    ui64 nRep = 1;
    double start, stop;

    ui64 z = RootOfUnity(opt::phim << 1, opt::q);
    ui64 z_p = RootOfUnity(opt::phim << 1, opt::p);
    ftt_precompute(z, opt::q, opt::logn);
    ftt_precompute(z_p, opt::p, opt::logn);
    encoding_precompute(opt::p, opt::logn);
    precompute_automorph_index(opt::phim);

    DiscreteGaussianGenerator dgg = DiscreteGaussianGenerator(4.0);

    FVParams slow_params{false,     opt::q,
                         opt::p,    opt::logn,
                         opt::phim, (opt::q / opt::p),
                         OPTIMIZED, make_shared<DiscreteGaussianGenerator>(dgg),
                         8};

    FVParams fast_params = slow_params;
    fast_params.fast_modulli = true;

    FVParams test_params = fast_params;

    //------------------- Synthetic Data -------------------
    uv64 vec_c = get_dgg_testvector(vec_size, opt::p);
    uv64 vec_s = get_dgg_testvector(vec_size, opt::p);

    //----------------------- KeyGen -----------------------
    nRep = 10;
    auto kp = KeyGen(test_params);

    start = currentDateTime();
    for (ui64 i = 0; i < nRep; i++) {
        kp = KeyGen(test_params);
    }
    stop = currentDateTime();
    cout << " KeyGen: " << (stop - start) / nRep << endl;

    //----------------- Client Preprocess ------------------
    nRep = 100;
    auto ct_vec = preprocess_client_share(kp.sk, vec_c, test_params);
    start = currentDateTime();
    for (ui64 i = 0; i < nRep; i++) {
        ct_vec = preprocess_client_share(kp.sk, vec_c, test_params);
    }
    stop = currentDateTime();
    cout << " Preprocess Client: " << (stop - start) / nRep << endl;

    //----------------- Server Preprocess -----------------
    vector<uv64> pt_vec;
    uv64 vec_s_f;
    tie(pt_vec, vec_s_f) = preprocess_server_share(vec_s, test_params);
    start = currentDateTime();
    for (ui64 i = 0; i < nRep; i++) {
        tie(pt_vec, vec_s_f) = preprocess_server_share(vec_s, test_params);
    }
    stop = currentDateTime();
    cout << " Preprocess Server: " << (stop - start) / nRep << endl;

    //---------------------- Square -----------------------
    auto ct_c_f = square_online(ct_vec, pt_vec, test_params);
    start = currentDateTime();
    for (ui64 i = 0; i < nRep; i++) {
        ct_c_f = square_online(ct_vec, pt_vec, test_params);
    }
    stop = currentDateTime();
    cout << " Multiply: " << (stop - start) / nRep << endl;

    //------------------- Post-Process ---------------------
    auto vec_c_f =
        postprocess_client_share(kp.sk, ct_c_f, vec_size, test_params);
    start = currentDateTime();
    for (ui64 i = 0; i < nRep; i++) {
        vec_c_f =
            postprocess_client_share(kp.sk, ct_c_f, vec_size, test_params);
    }
    stop = currentDateTime();
    cout << " Post-Process: " << (stop - start) / nRep << endl;

    //--------------------- Square PT ----------------------
    start = currentDateTime();
    auto vec_c_f_ref = square_pt(vec_c, vec_s, vec_s_f, opt::p);
    for (ui64 i = 0; i < nRep; i++) {
        vec_c_f_ref = square_pt(vec_c, vec_s, vec_s_f, opt::p);
    }
    stop = currentDateTime();
    cout << " Multiply PT: " << (stop - start) / nRep << endl;

    //----------------------- Check ------------------------
    cout << endl;

    check_vec_eq(vec_c_f_ref, vec_c_f, "square mismatch:\n");
}

void GRUCell::sigmoid(u64 n_circ) {
    vector<thread> thrds(2);
    thrds[0] = thread([n_circ]() { gc_sender(1213, 3, n_circ); });
    thrds[1] = thread([n_circ]() { gc_receiver(1213, 3, n_circ); });

    for (auto& thrd : thrds)
        thrd.join();
}

void GRUCell::tanh(u64 n_circ) {
    vector<thread> thrds(2);
    thrds[0] = thread([n_circ]() { gc_sender(1212, 4, n_circ); });
    thrds[1] = thread([n_circ]() { gc_receiver(1212, 4, n_circ); });

    for (auto& thrd : thrds)
        thrd.join();
}

void GRUCell::call() {
    matmul(m_input_size, 3 * m_hidden_size, 20);
    matmul(m_hidden_size, 3 * m_hidden_size, 20);
    elem_add(m_hidden_size);
    elem_mult(m_hidden_size);
    sigmoid(m_hidden_size);
    tanh(m_hidden_size);
}
}  // namespace lbcrypto