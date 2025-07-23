#ifndef SHA256_H
#define SHA256_H

#include <Arduino.h>
#include <Print.h>

#define HASH_LENGTH 32
#define BLOCK_LENGTH 64

class Sha256 : public Print {
public:
  void init(void);
  uint8_t* result(void);
  virtual size_t write(uint8_t);
  using Print::write;

private:
  void pad();
  void addUncounted(uint8_t data);
  void hashBlock();
  uint32_t rotl(uint32_t x, uint16_t n);
  uint32_t rotr(uint32_t x, uint16_t n);
  uint32_t Ch(uint32_t x, uint32_t y, uint32_t z);
  uint32_t Maj(uint32_t x, uint32_t y, uint32_t z);
  uint32_t Sigma0(uint32_t x);
  uint32_t Sigma1(uint32_t x);
  uint32_t sigma0(uint32_t x);
  uint32_t sigma1(uint32_t x);

  uint8_t buffer[BLOCK_LENGTH];
  uint32_t bufferOffset;
  uint32_t byteCount;
  uint32_t h[8];
};

// SHA-256 constants
static const uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

// Implementation
void Sha256::init(void) {
  h[0] = 0x6a09e667; h[1] = 0xbb67ae85; h[2] = 0x3c6ef372; h[3] = 0xa54ff53a;
  h[4] = 0x510e527f; h[5] = 0x9b05688c; h[6] = 0x1f83d9ab; h[7] = 0x5be0cd19;
  byteCount = 0; bufferOffset = 0;
}

uint32_t Sha256::rotl(uint32_t x, uint16_t n) { return (x << n) | (x >> (32 - n)); }
uint32_t Sha256::rotr(uint32_t x, uint16_t n) { return (x >> n) | (x << (32 - n)); }
uint32_t Sha256::Ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
uint32_t Sha256::Maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
uint32_t Sha256::Sigma0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
uint32_t Sha256::Sigma1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
uint32_t Sha256::sigma0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
uint32_t Sha256::sigma1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

void Sha256::hashBlock() {
  uint32_t a, b, c, d, e, f, g, h_loc, t1, t2, w[64];
  for (uint8_t i = 0; i < 16; i++) {
    w[i] = ((uint32_t)buffer[i * 4] << 24) | ((uint32_t)buffer[i * 4 + 1] << 16) | 
           ((uint32_t)buffer[i * 4 + 2] << 8) | buffer[i * 4 + 3];
  }
  for (uint8_t i = 16; i < 64; i++) {
    w[i] = sigma1(w[i - 2]) + w[i - 7] + sigma0(w[i - 15]) + w[i - 16];
  }
  a = h[0]; b = h[1]; c = h[2]; d = h[3]; e = h[4]; f = h[5]; g = h[6]; h_loc = h[7];
  for (uint8_t i = 0; i < 64; i++) {
    t1 = h_loc + Sigma1(e) + Ch(e, f, g) + k[i] + w[i];
    t2 = Sigma0(a) + Maj(a, b, c);
    h_loc = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += h_loc;
}

void Sha256::addUncounted(uint8_t data) {
  buffer[bufferOffset++] = data;
  if (bufferOffset == BLOCK_LENGTH) {
    hashBlock();
    bufferOffset = 0;
  }
}

size_t Sha256::write(uint8_t data) {
  ++byteCount;
  addUncounted(data);
  return 1;
}

void Sha256::pad() {
  addUncounted(0x80);
  while (bufferOffset != 56) {
    addUncounted(0x00);
    if (bufferOffset == 0) hashBlock();
  }
  uint64_t b = byteCount * 8;
  addUncounted(b >> 56); addUncounted(b >> 48); addUncounted(b >> 40); addUncounted(b >> 32);
  addUncounted(b >> 24); addUncounted(b >> 16); addUncounted(b >> 8); addUncounted(b);
  hashBlock();
}

uint8_t* Sha256::result(void) {
  pad();
  for (int i = 0; i < 8; i++) {
    for (int j = 3; j >= 0; j--) {
      buffer[i * 4 + (3 - j)] = (h[i] >> (j * 8)) & 0xFF;
    }
  }
  return buffer;
}

#endif
