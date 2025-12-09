# Sample Queries and Expected Responses

This document shows example queries you can ask the Pharmacy RAG Assistant and the expected behavior.

## Example 1: Side Effects Query

**Query:**
```
What are the side effects of aspirin?
```

**Expected Retrieval:**
- Top source: Aspirin (highest similarity score ~0.6-0.7)
- Additional sources: Ibuprofen, other NSAIDs (similar medications)

**Expected Response Format:**
```
Based on the provided information, aspirin has several common and serious side effects [S1]. 

Common side effects include:
- Stomach upset or heartburn
- Nausea
- Increased bleeding risk
- Bruising more easily

Serious side effects can include:
- Gastrointestinal bleeding
- Allergic reactions (rash, difficulty breathing)
- Reye's syndrome in children with viral infections
- Tinnitus (ringing in ears) with high doses [S1]

It's important to be aware of these risks, especially when taking aspirin with other 
medications like blood thinners [S1].
```

## Example 2: Storage Question

**Query:**
```
How should insulin be stored?
```

**Expected Retrieval:**
- Top source: Insulin (highest similarity score)
- Additional sources: Other medications with storage requirements

**Expected Response Format:**
```
Insulin storage is CRITICAL and requires specific temperature control [S1]. 

Storage requirements:
- Refrigerate unused insulin at 36-46°F (2-8°C)
- Do NOT freeze insulin
- Opened vials or pens can be kept at room temperature for 28 days

Proper storage is essential to maintain insulin effectiveness [S1].
```

## Example 3: Dosage Information

**Query:**
```
What is the recommended dosage for ibuprofen?
```

**Expected Retrieval:**
- Top source: Ibuprofen
- Additional sources: Similar pain medications

**Expected Response Format:**
```
Ibuprofen dosage varies by age and indication [S1]:

For adults:
- Typical dose: 200-400 mg every 4-6 hours as needed
- Maximum: 3200 mg/day for prescription strength

For children:
- Dosing based on weight: typically 5-10 mg/kg every 6-8 hours

Always follow the specific instructions provided by your healthcare provider [S1].
```

## Example 4: Drug Interactions

**Query:**
```
What medications interact with warfarin?
```

**Expected Retrieval:**
- Top source: Warfarin
- Additional sources: Aspirin, other blood thinners

**Expected Response Format:**
```
Warfarin has many important drug interactions that can affect blood clotting [S1]:

Medications that increase bleeding risk:
- Antibiotics
- NSAIDs (aspirin, ibuprofen)
- Acetaminophen
- Antifungals
- Other anticoagulants

Food interactions:
- Vitamin K-rich foods (leafy greens, broccoli) can affect warfarin levels

Always inform your healthcare provider about all medications and supplements you're 
taking when prescribed warfarin [S1].
```

## Example 5: Diabetes Medication

**Query:**
```
Tell me about metformin for diabetes
```

**Expected Retrieval:**
- Top source: Metformin
- Additional sources: Insulin, other diabetes medications

**Expected Response Format:**
```
Metformin is an oral diabetes medication used to control blood sugar in type 2 
diabetes [S1].

How it works:
- Decreases glucose production in the liver
- Improves insulin sensitivity

Typical dosage:
- Initial: 500 mg twice daily or 850 mg once daily with meals
- Maintenance: 1000-2000 mg daily in divided doses
- Maximum: 2550 mg/day

Common side effects (usually improve over time):
- Nausea
- Diarrhea
- Stomach upset
- Metallic taste

Serious but rare side effects:
- Lactic acidosis
- Vitamin B12 deficiency with long-term use [S1]
```

## Example 6: Multiple Drug Comparison

**Query:**
```
Compare aspirin and ibuprofen
```

**Expected Retrieval:**
- Top sources: Aspirin, Ibuprofen
- Both should rank highly

**Expected Response Format:**
```
Both aspirin and ibuprofen are NSAIDs used for pain and inflammation, but they have 
some key differences [S1, S2]:

Aspirin:
- Used for pain, fever, and cardiovascular protection
- Has antiplatelet effects for blood clot prevention
- Dosage: 325-650 mg every 4-6 hours for pain; 81-325 mg daily for heart protection
- Can cause Reye's syndrome in children [S1]

Ibuprofen:
- Used for pain, fever, and inflammation
- Does not have significant cardiovascular protection benefits
- Dosage: 200-400 mg every 4-6 hours
- Generally safer for children when dosed appropriately [S2]

Both can cause stomach upset and interact with blood thinners [S1, S2].
```

## Testing the System

### Retrieval Quality

The system should:
- ✅ Return relevant drug information for the query
- ✅ Rank most relevant source first (highest similarity score)
- ✅ Include related medications as additional context
- ✅ Handle variations in query phrasing

### Citation Quality

The system should:
- ✅ Include [S#] citations in the response
- ✅ Map citations to actual retrieved sources
- ✅ Use citations for specific facts and claims
- ✅ Not make up information beyond the sources

### Response Quality

The system should:
- ✅ Answer the specific question asked
- ✅ Provide concise, focused responses
- ✅ Include relevant details from the sources
- ✅ Stay within max_tokens limit (default 400)

## Edge Cases

### Query: General Health Question (Not in Database)

**Query:**
```
What causes high blood pressure?
```

**Expected Behavior:**
- System will retrieve most relevant sources
- Response should acknowledge limited information
- May provide related medication info (e.g., Lisinopril for treatment)

**Expected Response:**
```
While I don't have comprehensive information about the causes of high blood pressure 
in the database, I can share information about treatment. Lisinopril is an ACE 
inhibitor used to treat high blood pressure (hypertension) [S1]. It works by relaxing 
blood vessels to help lower blood pressure.

For detailed information about the causes of high blood pressure, please consult a 
healthcare provider.
```

### Query: Very Specific Detail

**Query:**
```
What is the molecular weight of aspirin?
```

**Expected Behavior:**
- System retrieves Aspirin information
- Response acknowledges information not in database
- Provides what information is available

### Query: Medication Not in Database

**Query:**
```
Tell me about lisinopril and atorvastatin combination
```

**Expected Behavior:**
- Retrieves both medications
- Provides information about each
- Notes they are both in the database

## Performance Expectations

### Retrieval Speed
- Query encoding: <1 second
- FAISS search: <1 second
- Total retrieval: <2 seconds

### Generation Speed (TinyLlama Q4)
- Context length 2048, max_tokens 400
- Generation: 10-30 seconds (CPU-dependent)
- Faster with fewer tokens or more CPU threads

### Resource Usage
- RAM: ~500MB-800MB (model + embeddings + FAISS)
- CPU: Varies with n_threads setting
- First run: +640MB for model download

## Quality Metrics

When testing, responses should demonstrate:
- **Accuracy**: Information matches source documents
- **Relevance**: Answers the specific question
- **Conciseness**: Stays within token limit
- **Citations**: Proper [S#] format
- **Safety**: Includes appropriate disclaimers

## Troubleshooting Responses

If responses are poor quality:

1. **Too generic**: Increase temperature (0.7 → 0.9)
2. **Too repetitive**: Decrease temperature (0.7 → 0.5)
3. **Too short**: Increase max_tokens (400 → 600)
4. **Off-topic**: Check retrieval quality, adjust top_k
5. **No citations**: Check prompt template

## Disclaimer

All responses should be understood as:
- ✅ Based on limited database information
- ✅ For informational purposes only
- ✅ Not a substitute for professional medical advice
- ✅ Generated by AI and may contain errors

Users should always consult healthcare professionals for medical decisions.
