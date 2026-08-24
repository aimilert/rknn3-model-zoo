#include "process.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <limits>
#include <vector>
#include <string>
#include <map>

int get_fbank_frames(knf::OnlineFbank *fbank, int frame_index, int num_frames, float *frames)
{
    if (fbank == nullptr || frames == nullptr)
    {
        fprintf(stderr, "Error: NULL parameter passed to get_fbank_frames\n");
        return -1;
    }

    if (frame_index + num_frames > fbank->NumFramesReady())
    {
        return -1;
    }

    for (int i = 0; i < num_frames; ++i)
    {
        const float *frame = fbank->GetFrame(i + frame_index);
        memcpy(frames + i * N_MELS, frame, N_MELS * sizeof(float));
    }

    return 0;
}

int argmax(float *array, int size)
{
    if (array == NULL || size <= 0)
    {
        return -1;
    }

    int max_index = 0;
    float max_value = array[0];
    for (int i = 1; i < size; i++)
    {
        if (array[i] > max_value)
        {
            max_value = array[i];
            max_index = i;
        }
    }
    return max_index;
}

float log_add(float a, float b)
{
    if (a == -std::numeric_limits<float>::infinity())
        return b;
    if (b == -std::numeric_limits<float>::infinity())
        return a;
    float mx = std::max(a, b);
    return mx + logf(expf(a - mx) + expf(b - mx));
}

void replace_substr(std::string &str, const std::string &from, const std::string &to)
{
    if (from.empty())
        return;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos)
    {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
}

int read_vocab(const char *fileName, VocabEntry *vocab)
{
    FILE *fp;
    char line[512];

    fp = fopen(fileName, "r");
    if (fp == NULL)
    {
        perror("Error opening vocab file");
        return -1;
    }

    int count = 0;
    while (fgets(line, sizeof(line), fp))
    {
        size_t line_len = strlen(line);
        if (line_len > 0 && line[line_len - 1] != '\n' && !feof(fp))
        {
            int c;
            while ((c = fgetc(fp)) != '\n' && c != EOF)
                ;
        }

        char *space_pos = strchr(line, ' ');
        if (space_pos == NULL)
        {
            fprintf(stderr, "Invalid line format in vocab file at line %d\n", count + 1);
            continue;
        }

        vocab[count].index = atoi(space_pos + 1);

        char *saveptr = NULL;
        char *token = strtok_r(line, " ", &saveptr);
        if (token != NULL)
        {
            vocab[count].token = strdup(token);
            if (vocab[count].token == NULL)
            {
                perror("Memory allocation failed for token");
                for (int i = 0; i < count; i++)
                {
                    if (vocab[i].token)
                    {
                        free(vocab[i].token);
                        vocab[i].token = NULL;
                    }
                }
                fclose(fp);
                return -1;
            }
        }

        count++;
    }

    fclose(fp);
    return 0;
}

std::vector<int> ctc_greedy_search(float *probs, int T, int V, int blank)
{
    std::vector<int> tokens;
    int prev = blank;
    for (int t = 0; t < T; t++)
    {
        int idx = argmax(probs + t * V, V);
        if (idx != blank && idx != prev)
        {
            tokens.push_back(idx);
        }
        prev = idx;
    }
    return tokens;
}

static bool prefix_score_cmp(const std::pair<std::vector<int>, PrefixScore> &a,
                             const std::pair<std::vector<int>, PrefixScore> &b)
{
    float sa = log_add(a.second.s, a.second.ns);
    float sb = log_add(b.second.s, b.second.ns);
    return sa > sb;
}

std::vector<CtcPrefix> ctc_prefix_beam_search(float *probs, int T, int V, int beam_size, int blank)
{
    float neg_inf = -std::numeric_limits<float>::infinity();

    // cur: prefix -> (blank_ending_score, non_blank_ending_score)
    std::map<std::vector<int>, PrefixScore> cur;
    cur[{}] = {0.0f, neg_inf};

    // precompute topk indices for each time step
    std::vector<int> indices(V);
    for (int i = 0; i < V; i++)
        indices[i] = i;

    for (int t = 0; t < T; t++)
    {
        float *logp = probs + t * V;

        // find top beam_size indices
        std::partial_sort(indices.begin(), indices.begin() + beam_size, indices.end(),
                          [logp](int a, int b)
                          { return logp[a] > logp[b]; });

        std::map<std::vector<int>, PrefixScore> nxt;

        for (int ki = 0; ki < beam_size; ki++)
        {
            int u = indices[ki];
            float p = logp[u];

            for (auto &[prefix, score] : cur)
            {
                float s = score.s;
                float ns = score.ns;
                int last = prefix.empty() ? -1 : prefix.back();

                if (u == blank)
                {
                    auto it = nxt.find(prefix);
                    if (it == nxt.end())
                        it = nxt.insert({prefix, {neg_inf, neg_inf}}).first;
                    it->second.s = log_add(it->second.s, log_add(s, ns) + p);
                }
                else if (u == last)
                {
                    // same token as last: update current prefix non-blank score
                    {
                        auto it = nxt.find(prefix);
                        if (it == nxt.end())
                            it = nxt.insert({prefix, {neg_inf, neg_inf}}).first;
                        it->second.ns = log_add(it->second.ns, ns + p);
                    }
                    // also extend with new prefix
                    {
                        std::vector<int> new_pf = prefix;
                        new_pf.push_back(u);
                        auto it = nxt.find(new_pf);
                        if (it == nxt.end())
                            it = nxt.insert({new_pf, {neg_inf, neg_inf}}).first;
                        it->second.ns = log_add(it->second.ns, s + p);
                    }
                }
                else
                {
                    std::vector<int> new_pf = prefix;
                    new_pf.push_back(u);
                    auto it = nxt.find(new_pf);
                    if (it == nxt.end())
                        it = nxt.insert({new_pf, {neg_inf, neg_inf}}).first;
                    it->second.ns = log_add(it->second.ns, log_add(s, ns) + p);
                }
            }
        }

        // keep top beam_size prefixes
        std::vector<std::pair<std::vector<int>, PrefixScore>> sorted_entries(nxt.begin(), nxt.end());
        std::sort(sorted_entries.begin(), sorted_entries.end(), prefix_score_cmp);
        if ((int)sorted_entries.size() > beam_size)
            sorted_entries.resize(beam_size);

        cur.clear();
        for (auto &entry : sorted_entries)
            cur[entry.first] = entry.second;
    }

    // build result
    std::vector<CtcPrefix> result;
    for (auto &[prefix, score] : cur)
    {
        CtcPrefix p;
        p.tokens = prefix;
        p.score = log_add(score.s, score.ns);
        result.push_back(p);
    }
    return result;
}
