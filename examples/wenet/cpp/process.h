#ifndef _RKNN_CONFORMER_DEMO_PROCESS_H_
#define _RKNN_CONFORMER_DEMO_PROCESS_H_

#include "rknn3_api.h"
#include "easy_timer.h"
#include "kaldi-native-fbank/csrc/online-feature.h"
#include <vector>
#include <string>
#include <map>

#define VOCAB_SIZE 4233
#define SAMPLE_RATE 16000
#define N_MELS 80
#define OUTPUT_SIZE 256
#define NUM_BLOCKS 12
#define HEAD 4
#define CNN_MODULE_KERNEL 8
#define SUBSAMPLING_RATE 4
#define RIGHT_CONTEXT 6
#define CHUNK_SIZE 16
#define LEFT_CHUNKS 4
#define ENCODER_OUT_LEN 200
#define BEAM_SIZE 10
#define MAX_HYP_LEN 13 //fix length, adjustable

#define DECODING_WINDOW ((CHUNK_SIZE - 1) * SUBSAMPLING_RATE + RIGHT_CONTEXT + 1) // 67
#define STRIDE (CHUNK_SIZE * SUBSAMPLING_RATE) // 64

#define BLANK_ID 0
#define SOS_ID 2
#define EOS_ID 2

#define REVERSE_WEIGHT 0.3f
#define CTC_WEIGHT 0.3f

// Check if VOCAB_PATH environment variable exists, otherwise use default
static inline const char *get_vocab_path()
{
    const char *env_path = getenv("VOCAB_PATH");
    if (env_path && env_path[0] != '\0')
    {
        return env_path;
    }
    return "./model/units.txt";
}

#define VOCAB_PATH get_vocab_path()

typedef struct
{
    int index;
    char *token;
} VocabEntry;

// CTC prefix beam search result
typedef struct
{
    std::vector<int> tokens;
    float score;
} CtcPrefix;

// Internal score for prefix beam search: (blank_ending, non_blank_ending)
struct PrefixScore
{
    float s;  // blank ending log-prob
    float ns; // non-blank ending log-prob
};

int get_fbank_frames(knf::OnlineFbank *fbank, int frame_index, int num_frames, float *frames);
int argmax(float *array, int size);
float log_add(float a, float b);
void replace_substr(std::string &str, const std::string &from, const std::string &to);
int read_vocab(const char *fileName, VocabEntry *vocab);
std::vector<int> ctc_greedy_search(float *probs, int T, int V, int blank = BLANK_ID);
std::vector<CtcPrefix> ctc_prefix_beam_search(float *probs, int T, int V, int beam_size = BEAM_SIZE, int blank = BLANK_ID);

#endif //_RKNN_CONFORMER_DEMO_PROCESS_H_
