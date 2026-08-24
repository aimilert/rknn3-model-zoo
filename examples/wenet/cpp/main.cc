/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> 
#include "conformer.h"
#include "audio_utils.h"
#include <iostream>
#include <vector>
#include <string>
#include "process.h"
#include <iomanip>

/*-------------------------------------------
                  Main Function
-------------------------------------------*/

int main(int argc, char **argv)
{
    if (argc < 8)
    {
        printf("Usage: %s <encoder_path> <encoder_weight> <ctc_path> <ctc_weight> "
               "<decoder_path> <decoder_weight> <audio_path>" 
               " [encoder_core_mask] [ctc_core_mask] [decoder_core_mask]\n",
               argv[0]);
        return -1;
    }

    const char *encoder_path = argv[1];
    const char *encoder_weight_path = argv[2];
    const char *ctc_path = argv[3];
    const char *ctc_weight_path = argv[4];
    const char *decoder_path = argv[5];
    const char *decoder_weight_path = argv[6];
    const char *audio_path = argv[7];
    uint32_t encoder_core_mask = (argc > 8) ? strtoul(argv[8], nullptr, 16) : 0xff;
    uint32_t ctc_core_mask     = (argc > 9) ? strtoul(argv[9], nullptr, 16) : 0xff;
    uint32_t decoder_core_mask = (argc > 10) ? strtoul(argv[10], nullptr, 16) : 0x1;

    // Validate file existence
    auto file_exists = [](const char *path) -> bool
    {
        FILE *fp = fopen(path, "r");
        if (fp)
        {
            fclose(fp);
            return true;
        }
        return false;
    };

    if (!file_exists(encoder_path))
    {
        printf("Error: Encoder model not found: %s\n", encoder_path);
        return -1;
    }
    if (!file_exists(encoder_weight_path))
    {
        printf("Error: Encoder weight not found: %s\n", encoder_weight_path);
        return -1;
    }
    if (!file_exists(ctc_path))
    {
        printf("Error: CTC model not found: %s\n", ctc_path);
        return -1;
    }
    if (!file_exists(ctc_weight_path))
    {
        printf("Error: CTC weight not found: %s\n", ctc_weight_path);
        return -1;
    }
    if (!file_exists(decoder_path))
    {
        printf("Error: Decoder model not found: %s\n", decoder_path);
        return -1;
    }
    if (!file_exists(decoder_weight_path))
    {
        printf("Error: Decoder weight not found: %s\n", decoder_weight_path);
        return -1;
    }
    if (!file_exists(audio_path))
    {
        printf("Error: Audio file not found: %s\n", audio_path);
        return -1;
    }

    int ret;
    TIMER timer;
    float infer_time = 0.0;
    float audio_length = 0.0;
    float rtf = 0.0;
    rknn_conformer_context_t rknn_app_ctx;
    VocabEntry vocab[VOCAB_SIZE];
    audio_buffer_t audio;
    std::string result_text;

    memset(&rknn_app_ctx, 0, sizeof(rknn_conformer_context_t));
    memset(vocab, 0, sizeof(vocab));
    memset(&audio, 0, sizeof(audio_buffer_t));

    // -- Read audio --
    timer.tik();
    ret = read_audio(audio_path, &audio);
    if (ret != 0)
    {
        printf("read audio fail! ret=%d\n", ret);
        goto out;
    }

    if (audio.num_channels == 2)
    {
        ret = convert_channels(&audio);
        if (ret != 0)
        {
            printf("convert channels fail! ret=%d\n", ret);
            goto out;
        }
    }

    if (audio.sample_rate != SAMPLE_RATE)
    {
        ret = resample_audio(&audio, audio.sample_rate, SAMPLE_RATE);
        if (ret != 0)
        {
            printf("resample audio fail! ret=%d\n", ret);
            goto out;
        }
    }

    ret = read_vocab(VOCAB_PATH, vocab);
    if (ret != 0)
    {
        printf("read vocab fail! ret=%d vocab_path=%s\n", ret, VOCAB_PATH);
        goto out;
    }
    timer.tok();
    timer.print_time("read_audio & preprocess & read_vocab");

    // -- Init encoder --
    timer.tik();
    ret = init_conformer_model(encoder_path, encoder_weight_path, &rknn_app_ctx.encoder_context, encoder_core_mask, CONFORMER_MODEL_ENCODER);
    if (ret != 0)
    {
        printf("init encoder fail! ret=%d\n", ret);
        goto out;
    }
    timer.tok();
    timer.print_time("init_encoder_model");

    // -- Init CTC --
    timer.tik();
    ret = init_conformer_model(ctc_path, ctc_weight_path, &rknn_app_ctx.ctc_context, ctc_core_mask, CONFORMER_MODEL_CTC);
    if (ret != 0)
    {
        printf("init ctc fail! ret=%d\n", ret);
        goto out;
    }
    timer.tok();
    timer.print_time("init_ctc_model");

    // -- Init decoder --
    timer.tik();
    ret = init_conformer_model(decoder_path, decoder_weight_path, &rknn_app_ctx.decoder_context, decoder_core_mask, CONFORMER_MODEL_DECODER);
    if (ret != 0)
    {
        printf("init decoder fail! ret=%d\n", ret);
        goto out;
    }
    timer.tok();
    timer.print_time("init_decoder_model");

    // -- Inference --
    timer.tik();
    ret = inference_conformer_model(&rknn_app_ctx, audio, vocab, result_text);
    if (ret != 0)
    {
        printf("inference fail! ret=%d\n", ret);
        goto out;
    }
    timer.tok();
    timer.print_time("inference_conformer_model");

    infer_time = timer.get_time() / 1000.0f; // sec
    audio_length = (float)audio.num_frames / audio.sample_rate;
    rtf = infer_time / audio_length;

    printf("\nReal Time Factor (RTF): %.3f / %.3f = %.3f\n", infer_time, audio_length, rtf);
    printf("\nConformer result: %s\n", result_text.c_str());

out:
    if (audio.data)
    {
        free(audio.data);
        audio.data = NULL;
    }

    for (int i = 0; i < VOCAB_SIZE; i++)
    {
        if (vocab[i].token)
        {
            free(vocab[i].token);
            vocab[i].token = NULL;
        }
    }

    release_conformer_model(&rknn_app_ctx.encoder_context);
    release_conformer_model(&rknn_app_ctx.ctc_context);
    release_conformer_model(&rknn_app_ctx.decoder_context);

    return 0;
}
