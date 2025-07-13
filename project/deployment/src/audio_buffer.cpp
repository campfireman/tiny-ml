#include "audio_buffer.hpp"

#include <cstring>

AudioBuffer::AudioBuffer(size_t maxSamples, size_t windowSamples)
    : m_ringSize(maxSamples),
      m_windowSize(windowSamples),
      m_ringBuf(new int32_t[m_ringSize]),
      m_winBuf(new int32_t[m_windowSize]),
      m_head(0)
{
}

// Caller writes directly into ringBuf at head…
int32_t *AudioBuffer::startFillSegment()
{
    return m_ringBuf + m_head;
}

// …then tells us how many samples were written:
void AudioBuffer::stopFillSegment(size_t segLen)
{
    // Advance head in ring buffer
    m_head = (m_head + segLen) % m_ringSize;

    // Slide window: drop oldest segLen, shift left
    const size_t keep = m_windowSize - segLen;
    std::memmove(m_winBuf,
                 m_winBuf + segLen,
                 keep * sizeof(int32_t));
    // Copy new segment (which just ended at head):
    size_t writePos = (m_head + m_ringSize - segLen) % m_ringSize;
    std::memcpy(m_winBuf + keep,
                m_ringBuf + writePos,
                segLen * sizeof(int32_t));
}

// Always returns contiguous window of length windowSize
int32_t *AudioBuffer::getLatestWindow()
{
    return m_winBuf;
}

size_t AudioBuffer::getWindowLength() const { return m_windowSize; }
