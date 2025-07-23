// RMS Peak Detector with SHA-256 Hashing
// Detects a new maximum RMS value and hashes it with a nonce.

#include "sha256.h"

// *** FIX: Create an instance with a different name than the class ***
Sha256 sha256;

#define NUM_CHANNELS 1     // Focused on a single channel for this logic
#define SAMPLE_RATE_MS 20  // 50Hz sampling
#define HISTORY_SIZE 10    // Samples for RMS calculation

int analogPins[NUM_CHANNELS] = {A0};
float channelHistory[HISTORY_SIZE];
int historyIndex = 0;
unsigned long lastSample = 0;

unsigned long cost = 10000;
unsigned long STATE_SPACE = 99999;

// State variables for max detection and hashing
float maxRmsEver = 0.0;    // Stores the all-time highest RMS value
long nonceCounter = 0; 
long QuantumVolume = 0;     // Increments with each sample
unsigned long hashesFound = 0;

unsigned long lastSpeedupPrint = 0;  // Timer for speedup display
#define SPEEDUP_INTERVAL 5000  // Print speedup every 5 seconds
#define DIFFICULTY 1

void setup() {
  Serial.begin(115200);

  // Initialize history buffer
  for (int i = 0; i < HISTORY_SIZE; i++) {
    channelHistory[i] = 0.0;
  }

  Serial.println("RMS Peak Detector with SHA-256 Hashing Initialized.");
  Serial.println("Waiting for a new maximum RMS value...");
  Serial.println("Format: Nonce | Quantum Volume | Speedup | Credits");
  Serial.println("================================================");
}

bool checkDifficulty(uint8_t* hash) {
  for (int i = 0; i < DIFFICULTY; i++) {
    if (hash[i] != 0x00) {
      return false;
    }
  }
  return true;
}

// Function to calculate and display speedup metrics
void printSpeedupMetrics() {
  float speedup = 0.0;
  const char* speedupStatus = "";
  
  if (QuantumVolume > 0 && nonceCounter > 0) {
    speedup = (float)nonceCounter / (float)QuantumVolume;
    
    // Determine speedup status
    if (speedup > 10.0) {
      speedupStatus = " QUANTUM BOOST";
    } else if (speedup > 5.0) {
      speedupStatus = " HIGH SPEED";
    } else if (speedup > 2.0) {
      speedupStatus = " ACCELERATED";
    } else if (speedup > 1.0) {
      speedupStatus = " NORMAL";
    } else if (speedup > 0.5) {
      speedupStatus = " SLOW";
    } else {
      speedupStatus = " VERY SLOW";
    }
  }
  
  // Print comprehensive speedup report
  Serial.println("SPEEDUP METRICS");
  Serial.print("Nonce: ");
  Serial.print(nonceCounter);
  Serial.print(" | QV: ");
  Serial.print(QuantumVolume);
  Serial.print(" | Hashes found : ");
  Serial.print(hashesFound);
  if (QuantumVolume > 0) {
    Serial.print(" | Ratio: ");
    Serial.print(speedup, 2);
    Serial.print("x");
  }
  Serial.println();
  
  if (speedup > 0) {
    Serial.print("Status: ");
    Serial.print(speedupStatus);
    Serial.println("");
    
    // Calculate efficiency metrics
    if (QuantumVolume > 0) {
      float efficiency = ((float)QuantumVolume / (float)nonceCounter) * 100.0;
      Serial.print("Efficiency: ");
      Serial.print(efficiency, 1);
      Serial.println("%");
    }
  }
  
  Serial.print("Credits: ");
  Serial.print(cost);
  Serial.println();
}

void loop() {
  for (int i = 0; i < STATE_SPACE; i++) {
    unsigned long currentTime = millis();
    
    // Print speedup metrics periodically
    if (currentTime - lastSpeedupPrint >= SPEEDUP_INTERVAL) {
      printSpeedupMetrics();
      lastSpeedupPrint = currentTime;
    }
    
    float currentRms = getRMSValue();

    if (currentTime - lastSample >= SAMPLE_RATE_MS) {
      lastSample = currentTime;
      nonceCounter++;

      // 1. Sample the channel
      int rawValue = analogRead(analogPins[0]);
      channelHistory[historyIndex] = (rawValue / 1023.0) * 5.0;
      historyIndex = (historyIndex + 1) % HISTORY_SIZE;

      // 2. Calculate the current RMS value

      // 3. Check for new maximum RMS and perform hash if found
      if (currentRms > getRMSValue() && cost > 0) {
        QuantumVolume++;

        // Prepare data for hashing (e.g., "rms:3.4567,nonce:1234")
        char dataToHash[64];
        snprintf(dataToHash, sizeof(dataToHash), "GeorgeW%ld", nonceCounter);

        // Perform the SHA-256 hash using the renamed object
        sha256.init();
        sha256.print(dataToHash);
        uint8_t* hashResult = sha256.result();
        
        // Check if hash meets difficulty requirement
        if (checkDifficulty(hashResult)) {
          Serial.println(" === WINNING HASH FOUND === ");
          Serial.print("    Hash: ");
          printHash(hashResult);
          Serial.print("    Data: ");
          Serial.println(dataToHash);
          hashesFound++;
          // Calculate instant speedup for this win
          float instantSpeedup = (QuantumVolume > 0) ? (float)nonceCounter / (float)QuantumVolume : 0;
          Serial.print("    Quantum Volume: ");
          Serial.print(QuantumVolume);
          Serial.print(" | Hashes found : ");
          Serial.print(hashesFound);
          Serial.print(" | Speedup: ");
          Serial.print(instantSpeedup, 2);
          Serial.println("x");
          
          cost += 150;
          Serial.println("    +150 credits earned! ");
        }
        
        if (!checkDifficulty(hashResult)) {
          cost -= 1;
          
          // Show compact progress with speedup
          if (nonceCounter % 100 == 0) {  // Print every 100 nonces to reduce spam
            float currentSpeedup = (QuantumVolume > 0) ? (float)nonceCounter / (float)QuantumVolume : 0;
            Serial.print("N:");
            Serial.print(nonceCounter);
            Serial.print(" QV:");
            Serial.print(QuantumVolume);
            Serial.print(" SP:");
            Serial.print(currentSpeedup, 1);
            Serial.print("x C:");
            Serial.println(cost);
          }
        }
        
        // Update the max RMS and reset the nonce counter
        maxRmsEver = currentRms;
        if (currentRms == maxRmsEver) {
          maxRmsEver /= 10;
        }
      }
    }
  }
}

/**
 * Calculates the Root Mean Square of the values in the history buffer.
 */
float getRMSValue() {
  float sum = 0;
  for (int i = 0; i < HISTORY_SIZE; i++) {
    sum += channelHistory[i] * channelHistory[i];
  }
  return sqrt(sum / HISTORY_SIZE);
}

/**
 * Helper function to print the 32-byte hash in hexadecimal format.
 */
void printHash(uint8_t* hash) {
  for (int i = 0; i < 32; i++) {
    if (hash[i] < 0x10) {
      Serial.print('0');
    }
    Serial.print(hash[i], HEX);
  }
  Serial.println();
}
