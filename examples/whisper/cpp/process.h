#ifndef _RKNN_WHISPER_DEMO_PROCESS_H_
#define _RKNN_WHISPER_DEMO_PROCESS_H_

// #define TIMING_DISABLED // if you don't need to print the time used, uncomment this line of code

#include <string>
#include <vector>
#include <cstdint>

#include "audio_utils.h"

#define VOCAB_NUM 51865
#define SAMPLE_RATE 16000
#define N_FFT 400
#define HOP_LENGTH 160
#define CHUNK_LENGTH 30
#define MAX_AUDIO_LENGTH CHUNK_LENGTH *SAMPLE_RATE
#define N_MELS 80
#define MELS_FILTERS_SIZE 201 // (N_FFT / 2 + 1)


#define MEL_FILTERS_PATH "./model/mel_80_filters.txt"


typedef struct
{
    int index;
    char *token;
} VocabEntry;

void replace_substr(std::string &str, const std::string &from, const std::string &to);
int read_vocab(const char *fileName, VocabEntry *vocab);
int read_mel_filters(const char *fileName, float *data, int max_lines);
void audio_preprocess(audio_buffer_t *audio, float *mel_filters, std::vector<float> &x_mel);
std::string normalize_piece(const std::string &piece);
std::string decode_whisper_tokens(VocabEntry *vocab, const std::vector<int32_t> &token_ids, const char *task);
std::string base64_decode(const std::string &s);

#endif //_RKNN_WHISPER_DEMO_PROCESS_H_
