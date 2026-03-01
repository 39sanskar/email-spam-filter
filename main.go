package main

import (
	"fmt"
	"io/fs"
	"io/ioutil"
	"math"
	"path/filepath"
	"strings"
)

// THRESHOLD is used to ignore tokens that occur less than this number across all emails.
const THRESHOLD = 300

// Bow represents a bag-of-words: word -> frequency mapping
type Bow map[string]int

// tokenize converts message into uppercase tokens split by whitespace
func tokenize(message string) []string {
	tokens := strings.Fields(message)
	for i := range tokens {
		tokens[i] = strings.ToUpper(strings.TrimSpace(tokens[i]))
	}
	return tokens
}

// addFileToBow reads a file and adds its tokens to the given Bag-of-Words map
func addFileToBow(filePath string, bow Bow) error {
	
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to read file %s: %w", filePath, err)
	}

	for _, token := range tokenize(string(content)) {
		bow[token]++
	}
	return nil
}

// addDirToBow walks through directory recursively and builds BoW from all .txt files
func addDirToBow(dirPath string, bow Bow) error {
	// WalkDir(root string, fn fs.WalkDirFunc) error
	return filepath.WalkDir(dirPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		return addFileToBow(path, bow)
	})
}

// totalCount sums frequencies of only those words which appear at least THRESHOLD times
func totalCount(bow Bow) int {
	count := 0
	for _, freq := range bow {
		if freq >= THRESHOLD {
			count += freq
		}
	}
	return count
}

// classifyFile returns log-probabilities for being spam or ham based on trained models
func classifyFile(hamBow Bow, hamTotalCount int, spamBow Bow, spamTotalCount int, filePath string) (float64, float64, error) {
	totalDocCount := hamTotalCount + spamTotalCount

	fileBow := make(Bow)
	if err := addFileToBow(filePath, fileBow); err != nil {
		return 0, 0, err
	}

	// Initial prior probabilities (log space)
	hamPriorLogProb := math.Log(float64(hamTotalCount) / float64(totalDocCount))
	spamPriorLogProb := math.Log(float64(spamTotalCount) / float64(totalDocCount))

	var hamLogLikelihood float64 = 0
	var spamLogLikelihood float64 = 0
	var docLogLikelihood float64 = 0

	for token := range fileBow {
		tokenFreqInHam := hamBow[token]
		tokenFreqInSpam := spamBow[token]

		// Only consider common-enough words
		if tokenFreqInHam+tokenFreqInSpam < THRESHOLD {
			continue
		}

		// Add likelihoods in log-space
		if tokenFreqInHam > 0 {
			hamLogLikelihood += math.Log(float64(tokenFreqInHam) / float64(hamTotalCount))
		}
		if tokenFreqInSpam > 0 {
			spamLogLikelihood += math.Log(float64(tokenFreqInSpam) / float64(spamTotalCount))
		}

		docLogLikelihood += math.Log(float64(tokenFreqInHam+tokenFreqInSpam) / float64(totalDocCount))
	}

	// Return posterior scores (simplified as difference from document likelihood)
	hamScore := hamPriorLogProb + hamLogLikelihood - docLogLikelihood
	spamScore := spamPriorLogProb + spamLogLikelihood - docLogLikelihood

	return spamScore, hamScore, nil
}

// classifyDir classifies each file in a directory and counts predictions
func classifyDir(hamBow Bow, hamTotalCount int, spamBow Bow, spamTotalCount int, dirPath string) (int, int, error) {
	spamPredictionCount := 0
	hamPredictionCount := 0

	err := filepath.WalkDir(dirPath, func(filePath string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil || d.IsDir() {
			return nil
		}

		spamScore, hamScore, err := classifyFile(hamBow, hamTotalCount, spamBow, spamTotalCount, filePath)
		if err != nil {
			return err
		}

		if spamScore > hamScore {
			spamPredictionCount++
		} else {
			hamPredictionCount++
		}
		return nil
	})

	return spamPredictionCount, hamPredictionCount, err
}

// main trains on Enron datasets and evaluates performance on test data
func main() {
	fmt.Println("Training...")

	hamBow := make(Bow)
	spamBow := make(Bow)

	// Train over multiple dataset folders
	for i := 1; i <= 5; i++ {
		hamPath := fmt.Sprintf("./enron%d/ham", i)
		spamPath := fmt.Sprintf("./enron%d/spam", i)

		if err := addDirToBow(hamPath, hamBow); err != nil {
			panic(fmt.Sprintf("Error reading ham dir %s: %v", hamPath, err))
		}
		if err := addDirToBow(spamPath, spamBow); err != nil {
			panic(fmt.Sprintf("Error reading spam dir %s: %v", spamPath, err))
		}
	}

	// Total word counts above threshold
	hamTotalCount := totalCount(hamBow)
	spamTotalCount := totalCount(spamBow)

	fmt.Println("Classifying ham...")
	spamOutcomeCount, hamOutcomeCount, err := classifyDir(hamBow, hamTotalCount, spamBow, spamTotalCount, "./enron5/ham")
	if err != nil {
		panic(fmt.Sprintf("Failed to classify ham: %v", err))
	}
	fmt.Printf("Ham classified:\n  Spam predictions = %d\n  Ham predictions = %d\n", spamOutcomeCount, hamOutcomeCount)

	fmt.Println("\nClassifying spam...")
	spamOutcomeCount, hamOutcomeCount, err = classifyDir(hamBow, hamTotalCount, spamBow, spamTotalCount, "./enron5/spam")
	if err != nil {
		panic(fmt.Sprintf("Failed to classify spam: %v", err))
	}
	fmt.Printf("Spam classified:\n  Spam predictions = %d\n  Ham predictions = %d\n", spamOutcomeCount, hamOutcomeCount)
}


