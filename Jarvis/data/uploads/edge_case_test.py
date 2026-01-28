from preprocess import get_disability
import time

def test_edge_cases():
    """Test edge cases with long sentences and various typos"""
    
    print("=== EDGE CASE TESTING ===\n")
    
    # Test cases with one-letter typos in long sentences
    test_cases = [
        # Long medical descriptions with single typos
        {
            "sentence": "Autism Spectrum Disorder with sevire communication deficits and repetitive behaviors",
            "typo": "sevire (should be severe)"
        },
        {
            "sentence": "Intellectual Disabilty, Moderate level with adaptive behavior challenges and learning difficulties",
            "typo": "Disabilty (should be Disability)"
        },
        {
            "sentence": "Attention Deficit Hyperactivity Disorder with impulsivity, inattention, and hyperactive behavor patterns",
            "typo": "behavor (should be behavior)"
        },
        {
            "sentence": "Borderline Intellectual Functioning with cognitive processing delays and acadmic struggles",
            "typo": "acadmic (should be academic)"
        },
        {
            "sentence": "Down Syndrome with developmental delays, cognitive impairment, and characteristic physical featurs",
            "typo": "featurs (should be features)"
        },
        {
            "sentence": "Cerebral Palsy affecting motor function, posture, and movment coordination with spastic symptoms",
            "typo": "movment (should be movement)"
        },
        {
            "sentence": "Developmental Delay in multiple domains including cognitive, motor, and social-emotional developmnt",
            "typo": "developmnt (should be development)"
        },
        {
            "sentence": "Learning Disability affecting reading comprehension, mathematical reasoning, and writen expression",
            "typo": "writen (should be written)"
        },
        
        # Test with multiple words but single typo
        {
            "sentence": "Unspecified Intellectual Disability with mild to moderat cognitive impairment and adaptive behavior deficits",
            "typo": "moderat (should be moderate)"
        },
        {
            "sentence": "Pervasive Developmental Disorder Not Otherwise Specified with social communication challanges",
            "typo": "challanges (should be challenges)"
        },
        
        # Test very long sentences
        {
            "sentence": "Autism Spectrum Disorder Level 2 requiring substantial support with restricted, repetitive patterns of behavior, interests, or activities that cause clinically significant impairment in social, occupational, or other important areas of current functionng",
            "typo": "functionng (should be functioning)"
        },
        {
            "sentence": "Intellectual Disability, Severe, with profound limitations in adaptive behavior across multiple environments including home, school, work, and community settings that require extensive ongoing suport",
            "typo": "suport (should be support)"
        },
        
        # Test edge cases
        {
            "sentence": "ADHD",  # Very short
            "typo": "None"
        },
        {
            "sentence": "Mild Intellectual Dsability",  # Simple case
            "typo": "Dsability (should be Disability)"
        },
        {
            "sentence": "",  # Empty string
            "typo": "Empty"
        },
        {
            "sentence": "nan",  # NaN case
            "typo": "NaN value"
        }
    ]
    
    print(f"Testing {len(test_cases)} edge cases with various sentence lengths and typos...\n")
    
    total_time = 0
    for i, test_case in enumerate(test_cases, 1):
        sentence = test_case["sentence"]
        typo_info = test_case["typo"]
        
        print(f"Test {i}: {typo_info}")
        print(f"Original: '{sentence}'")
        
        # Time the processing
        start_time = time.time()
        result = get_disability(sentence, [])
        processing_time = time.time() - start_time
        total_time += processing_time
        
        print(f"Result:   '{result}'")
        print(f"Time:     {processing_time:.3f} seconds")
        print("-" * 80)
    
    print(f"\nSUMMARY:")
    print(f"Total tests: {len(test_cases)}")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Average time per test: {total_time/len(test_cases):.3f} seconds")
    
    # Test batch processing for comparison
    print(f"\n=== BATCH PROCESSING COMPARISON ===")
    sentences = [case["sentence"] for case in test_cases if case["sentence"] and case["sentence"] != "nan"]
    
    start_time = time.time()
    batch_results = []
    for sentence in sentences:
        result = get_disability(sentence, [])
        batch_results.append(result)
    batch_time = time.time() - start_time
    
    print(f"Processed {len(sentences)} sentences")
    print(f"Batch time: {batch_time:.3f} seconds")
    print(f"Average per sentence: {batch_time/len(sentences):.3f} seconds")

if __name__ == "__main__":
    test_edge_cases()
