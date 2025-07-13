// AudioBuffer.h

#ifndef AUDIOBUFFER_H
#define AUDIOBUFFER_H

#include <cstdint>
#include <cstddef>

class AudioBuffer
{
public:
    AudioBuffer(size_t maxSamples, size_t windowSamples);

    ~AudioBuffer();

    int32_t *startFillSegment();

    void stopFillSegment(size_t segLen);

    int32_t *getLatestWindow();

    size_t getWindowLength() const;

    AudioBuffer(const AudioBuffer &) = delete;
    AudioBuffer &operator=(const AudioBuffer &) = delete;

private:
    size_t m_ringSize;
    size_t m_windowSize;
    int32_t *m_ringBuf;
    int32_t *m_winBuf;
    size_t m_head;
};

#endif // AUDIOBUFFER_H