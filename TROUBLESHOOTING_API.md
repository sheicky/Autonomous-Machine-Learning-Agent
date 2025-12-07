# 🔧 API Troubleshooting Guide

## Common OpenRouter API Issues & Solutions

### Issue 1: Rate Limit (429 Error)

**Symptoms:**
```
Rate limit (429). Waiting 44.5s...
Rate limit (429). Waiting 83.1s...
```

**Causes:**
- Too many requests in short time
- Free tier rate limits exceeded
- Multiple users sharing same API key

**Solutions:**

#### Quick Fix (Immediate)
The enhanced version now automatically tries multiple models:
1. Your selected model
2. Llama 3.3 70B (free, higher limits)
3. Gemini 2.0 Flash (free)
4. Qwen 2.5 7B (free)

**No action needed** - it will automatically switch models!

#### Manual Fix (If needed)
Change model in sidebar:
```
Sidebar → LLM Model → Select "Gemini 2.0 Flash"
```

#### Long-term Fix
1. **Get API key with higher limits:**
   - Visit https://openrouter.ai
   - Add credits to your account
   - Higher limits with paid tier

2. **Use caching (automatic):**
   - Responses are cached automatically
   - Subsequent runs use cache
   - No API calls for repeated operations

---

### Issue 2: Timeout Error

**Symptoms:**
```
HTTPSConnectionPool(host='openrouter.ai', port=443): Read timed out. (read timeout=120)
```

**Causes:**
- Model is slow to respond
- Network issues
- Server overload

**Solutions:**

#### Automatic (Already Implemented)
- Timeout increased to 180 seconds
- Automatically tries next model on timeout
- Max 3 attempts per model

#### Manual Workarounds
1. **Try smaller model:**
   ```
   Sidebar → LLM Model → "Qwen 2.5 7B"
   ```

2. **Check internet connection:**
   ```bash
   ping openrouter.ai
   ```

3. **Wait and retry:**
   - Server might be overloaded
   - Try again in 5-10 minutes

---

### Issue 3: All Models Failed

**Symptoms:**
```
[ERROR] All LLM models failed. Last error: ...
```

**Causes:**
- API key invalid
- Network completely down
- OpenRouter service outage

**Solutions:**

#### Check API Key
```bash
# In app/.env
OPENROUTER_API_KEY=sk-or-v1-...  # Should start with sk-or-v1-
```

#### Verify API Key Works
```bash
curl https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

#### Check OpenRouter Status
Visit: https://status.openrouter.ai

#### Alternative: Use Local LLM (Advanced)
If OpenRouter is down, you can modify the code to use local models:
- Ollama
- LM Studio
- LocalAI

---

### Issue 4: Slow Response Times

**Symptoms:**
- Takes 2-3 minutes per LLM call
- Multiple retries happening

**Solutions:**

#### Use Faster Models
```
Recommended order (fastest to slowest):
1. Qwen 2.5 7B (fastest, free)
2. Gemini 2.0 Flash (fast, free)
3. Llama 3.3 70B (slower, free)
```

#### Enable Caching (Automatic)
- First run: Slow (makes API calls)
- Second run: Fast (uses cache)
- Cache location: In-memory

#### Reduce Complexity
- Use smaller datasets for testing
- Skip quality analysis if not needed
- Use fewer models for training

---

## Best Practices

### 1. Start with Fastest Model
```python
# In app/.env
OPENROUTER_MODEL=qwen/qwen-2.5-7b-instruct:free
```

### 2. Use Caching
- Don't clear cache unnecessarily
- Cache is automatic
- Saves API calls and time

### 3. Batch Operations
- Upload data once
- Run full pipeline
- Don't restart unnecessarily

### 4. Monitor Usage
Check your OpenRouter dashboard:
- https://openrouter.ai/activity
- See request count
- Check remaining credits

### 5. Upgrade if Needed
Free tier limits:
- ~20 requests/minute
- ~200 requests/day

Paid tier:
- Much higher limits
- Faster response
- Priority access

---

## Error Messages Explained

### "Rate limit (429)"
**Meaning:** Too many requests  
**Action:** Wait or switch model (automatic)

### "Read timed out"
**Meaning:** Response took too long  
**Action:** Try next model (automatic)

### "Connection refused"
**Meaning:** Can't reach server  
**Action:** Check internet, try later

### "Invalid API key"
**Meaning:** Wrong or expired key  
**Action:** Check .env file

### "Model not found"
**Meaning:** Model name incorrect  
**Action:** Use recommended models

---

## Quick Diagnostics

### Test 1: Check API Key
```bash
# Should return list of models
curl https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY" | head -20
```

### Test 2: Check Network
```bash
ping openrouter.ai
# Should get responses
```

### Test 3: Test Simple Request
```python
import requests
import os

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    },
    json={
        "model": "qwen/qwen-2.5-7b-instruct:free",
        "messages": [{"role": "user", "content": "Say hello"}]
    },
    timeout=30
)
print(response.status_code)
print(response.json())
```

---

## Recommended Settings

### For Speed
```
Model: Qwen 2.5 7B
Timeout: 180s (default)
Retries: 3 per model (default)
```

### For Quality
```
Model: Llama 3.3 70B
Timeout: 180s (default)
Retries: 3 per model (default)
```

### For Reliability
```
Model: Auto (tries all models)
Timeout: 180s (default)
Retries: 3 per model (default)
```

---

## What's Improved in Enhanced Version

✅ **Automatic Model Fallback**
- Tries 4 models automatically
- No manual intervention needed

✅ **Smarter Retry Logic**
- Max 3 attempts per model
- Caps wait time at 60s
- Switches models on timeout

✅ **Better Error Messages**
- Shows which model failed
- Shows which model succeeded
- Clear error descriptions

✅ **Increased Timeout**
- 120s → 180s
- Handles slower models better

✅ **Response Size Limit**
- Max 4000 tokens
- Prevents huge responses
- Faster processing

---

## Still Having Issues?

### Option 1: Use Cached Responses
If you've run successfully before, the cache will work without API calls.

### Option 2: Try Later
OpenRouter free tier can be busy during peak hours.

### Option 3: Upgrade API Key
Get higher limits: https://openrouter.ai/credits

### Option 4: Use Alternative
- Try Gemini API directly (free)
- Try OpenAI API (paid)
- Use local LLM (Ollama)

---

## Contact Support

**OpenRouter Issues:**
- https://openrouter.ai/docs
- Discord: https://discord.gg/openrouter

**App Issues:**
- Check GitHub issues
- Review documentation
- Run diagnostics above

---

**Last Updated:** December 2024  
**Status:** Enhanced version includes automatic fallback ✅
