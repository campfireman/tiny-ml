#include "audio_buffer.hpp"

#include <cstring>
#include <algorithm>

AudioBuffer::AudioBuffer(size_t maxSamples, size_t segLen, size_t windowSamples)
    : m_ringSize(maxSamples),
      m_segLen(segLen),
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
void AudioBuffer::stopFillSegment()
{
    // Advance head in ring buffer
    m_head = (m_head + m_segLen) % m_ringSize;
}

// Always returns contiguous window of length windowSize
int32_t *AudioBuffer::getLatestWindow()
{
    size_t tail = (m_head + m_ringSize - m_windowSize) % m_ringSize;
    size_t first = std::min(m_windowSize, m_ringSize - tail);
    std::memcpy(m_winBuf, m_ringBuf + tail, first * sizeof(int32_t));
    if (m_windowSize > first)
    {
        std::memcpy(m_winBuf + first, m_ringBuf, (m_windowSize - first) * sizeof(int32_t));
    }
    return m_winBuf;
}

size_t AudioBuffer::getWindowLength() const { return m_windowSize; }
