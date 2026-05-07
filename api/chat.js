module.exports = async (req, res) => {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    const { messages, provider, model, mode, prompt, imageDataUrl, imageMimeType, internetSearch, url, analysisType } = req.body;
    const isInternetSearch = Boolean(internetSearch);

    // Handle image generation mode
    if (mode === 'image') {
        if (provider !== 'gemini') {
            return res.status(400).json({ error: 'Image generation only supported with Gemini provider' });
        }
        if (!prompt) {
            return res.status(400).json({ error: 'Missing prompt for image generation' });
        }
        const geminiApiKey = process.env.GEMINI_API_KEY;
        if (!geminiApiKey) {
            return res.status(400).json({ error: 'Missing GEMINI_API_KEY environment variable' });
        }
        try {
            const contents = [{
                role: 'user',
                parts: [{ text: prompt }]
            }];
            const geminiResponse = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
                method: 'POST',
                headers: {
                    'x-goog-api-key': geminiApiKey,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ contents })
            });
            if (!geminiResponse.ok) {
                const errorData = await geminiResponse.json().catch(() => ({}));
                const errorMsg = errorData.error?.message || geminiResponse.statusText;
                const fullErrorContext = JSON.stringify(errorData) + ' ' + errorMsg;
                // Kontrollera om det är ett kvotfel för gratisnivån
                const isQuotaExhausted = fullErrorContext.includes('limit: 0') || 
                                        fullErrorContext.includes('generate_content_free_tier_requests') || 
                                        fullErrorContext.includes('Quota exceeded');
                if (isQuotaExhausted) {
                    return res.status(403).json({ 
                        error: 'Gemini image generation is not available on your current free-tier quota. Enable billing/Tier 1 in Google AI Studio or use another image provider.' 
                    });
                }
                // Hantera normala rate limit-fel (429 utan kvotproblem)
                if (geminiResponse.status === 429) {
                    return res.status(429).json({ 
                        error: 'Gemini is rate limited. Try again later.' 
                    });
                }
                // Andra fel
                return res.status(geminiResponse.status).json({ error: `Gemini image generation error: ${errorMsg}` });
            }
            const geminiData = await geminiResponse.json();
            let text = '';
            let imageDataUrl = null;
            const candidate = geminiData.candidates?.[0];
            if (candidate && candidate.content && candidate.content.parts) {
                for (const part of candidate.content.parts) {
                    if (part.text) {
                        text += part.text;
                    }
                    if (part.inline_data) {
                        const { mime_type, data } = part.inline_data;
                        imageDataUrl = `data:${mime_type};base64,${data}`;
                    }
                }
            }
            if (!text && !imageDataUrl) {
                text = 'No image generated.';
            }
            return res.status(200).json({
                text: text || null,
                imageDataUrl,
                providerUsed: 'gemini',
                modelUsed: model,
                mode: 'image',
                internetSearchUsed: false,
                sources: null
            });
        } catch (error) {
            console.error('Image generation error:', error);
            return res.status(500).json({ error: `Image generation failed: ${error.message}` });
        }
    }

    // Handle analyze URL mode
    if (mode === 'analyze_url') {
        if (!url || !analysisType) {
            return res.status(400).json({ error: 'Missing url or analysisType' });
        }
        if (!url.startsWith('http://') && !url.startsWith('https://')) {
            return res.status(400).json({ error: 'Invalid URL. Must start with http:// or https://' });
        }
        const selectedProvider = provider || 'openrouter';
        const selectedModel = model || 'tencent/hy3-preview:free';

        // Get user message if provided
        const userMsg = (messages && messages.length > 0 && messages[0].role === 'user') ? messages[0].content : null;

        try {
            // Fetch the webpage with timeout
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 10000);
            let fetchResponse;
            try {
                fetchResponse = await fetch(url, { 
                    signal: controller.signal,
                    headers: { 'User-Agent': 'Rene.Gpt Analyzer' }
                });
            } catch (fetchError) {
                clearTimeout(timeout);
                if (fetchError.name === 'AbortError') {
                    return res.status(400).json({ error: 'Fetch timeout: Could not fetch webpage within 10 seconds.' });
                }
                return res.status(400).json({ error: `Could not fetch webpage: ${fetchError.message}` });
            }
            clearTimeout(timeout);

            if (!fetchResponse.ok) {
                return res.status(400).json({ error: `Could not fetch webpage: ${fetchResponse.status} ${fetchResponse.statusText}` });
            }

            let html = await fetchResponse.text();
            // Limit HTML size to ~1MB
            if (html.length > 1000000) {
                html = html.substring(0, 1000000);
            }

            // Extract SEO data
            const seoData = extractSeoData(html, url);

            // Build analysis prompt
            const promptText = buildAnalysisPrompt(url, analysisType, seoData, userMsg);

            // Prepare messages for AI
            const analysisMessages = [{ role: 'user', content: promptText }];

            // Call AI based on provider
            let aiResponseText = '';
            let usedProvider = selectedProvider;
            let usedModel = selectedModel;

            if (selectedProvider === 'openrouter') {
                const openrouterApiKey = process.env.OPENROUTER_API_KEY;
                if (!openrouterApiKey) {
                    return res.status(400).json({ error: 'Missing OPENROUTER_API_KEY environment variable' });
                }
                const openrouterResponse = await fetch('https://openrouter.ai/api/v1/chat/completions', {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${openrouterApiKey}`,
                        'Content-Type': 'application/json',
                        'HTTP-Referer': req.headers.origin || 'http://localhost',
                        'X-Title': 'rene.gpt'
                    },
                    body: JSON.stringify({
                        model: selectedModel,
                        messages: analysisMessages
                    })
                });
                if (!openrouterResponse.ok) {
                    const errorData = await openrouterResponse.json().catch(() => ({}));
                    const errorMsg = errorData.error?.message || openrouterResponse.statusText;
                    return res.status(openrouterResponse.status).json({ error: `OpenRouter error: ${errorMsg}` });
                }
                const data = await openrouterResponse.json();
                aiResponseText = data.choices?.[0]?.message?.content || 'No analysis.';
                usedProvider = 'openrouter';
                usedModel = selectedModel;
            } else if (selectedProvider === 'gemini') {
                const geminiApiKey = process.env.GEMINI_API_KEY;
                if (!geminiApiKey) {
                    return res.status(400).json({ error: 'Missing GEMINI_API_KEY environment variable' });
                }
                const geminiContents = convertMessagesToGemini(analysisMessages);
                const geminiResponse = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${selectedModel}:generateContent`, {
                    method: 'POST',
                    headers: {
                        'x-goog-api-key': geminiApiKey,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ contents: geminiContents })
                });
                if (!geminiResponse.ok) {
                    const errorData = await geminiResponse.json().catch(() => ({}));
                    const errorMsg = errorData.error?.message || geminiResponse.statusText;
                    return res.status(geminiResponse.status).json({ error: `Gemini error: ${errorMsg}` });
                }
                const geminiData = await geminiResponse.json();
                aiResponseText = geminiData.candidates?.[0]?.content?.parts?.[0]?.text || 'No analysis.';
                usedProvider = 'gemini';
                usedModel = selectedModel;
            } else {
                return res.status(400).json({ error: `Unknown provider: ${selectedProvider}` });
            }

            return res.status(200).json({
                text: aiResponseText,
                providerUsed: usedProvider,
                modelUsed: usedModel,
                mode: 'analyze_url',
                url,
                analysisType,
                extractedSeoData: seoData
            });

        } catch (error) {
            console.error('Analyze URL error:', error);
            return res.status(500).json({ error: `Analyze URL failed: ${error.message}` });
        }
    }

    // --- Chat mode (existing logic) ---
    if (!messages || !provider || !model) {
        return res.status(400).json({ error: 'Missing required fields: messages, provider, model' });
    }

    const openrouterApiKey = process.env.OPENROUTER_API_KEY;
    const geminiApiKey = process.env.GEMINI_API_KEY;

    let fallbackUsed = false;
    let originalProviderError = null;
    let providerUsed = provider;
    let modelUsed = model;
    let responseText = '';
    let sources = null;
    let internetSearchUsed = isInternetSearch;

    // Validate image if present
    if (imageDataUrl) {
        // Check image data URL format
        if (!imageDataUrl.startsWith('data:image/')) {
            return res.status(400).json({ error: 'Invalid image data URL: must start with data:image/' });
        }

        // Check image size (4MB base64 string limit)
        if (imageDataUrl.length > 4 * 1024 * 1024) {
            return res.status(400).json({ error: 'Image is too large. Try a smaller image.' });
        }

        // Check MIME type
        if (!imageMimeType || !imageMimeType.startsWith('image/')) {
            return res.status(400).json({ error: 'Invalid image MIME type' });
        }
    }

    try {
        if (provider === 'openrouter') {
            // Validate OpenRouter API key
            if (!openrouterApiKey) {
                return res.status(400).json({ error: 'Missing OPENROUTER_API_KEY environment variable' });
            }

            // Attempt OpenRouter request
            try {
                // Prepare messages for OpenRouter
                let openrouterMessages = messages.map(msg => ({ ...msg }));

                // If image is present, modify last user message to include image
                if (imageDataUrl) {
                    const lastUserMsgIndex = openrouterMessages.slice().reverse().findIndex(msg => msg.role === 'user');
                    if (lastUserMsgIndex === -1) {
                        return res.status(400).json({ error: 'No user message found to attach image to' });
                    }
                    const actualIndex = openrouterMessages.length - 1 - lastUserMsgIndex;
                    const userText = openrouterMessages[actualIndex].content || '';
                    openrouterMessages[actualIndex] = {
                        role: 'user',
                        content: [
                            { type: 'text', text: userText },
                            { type: 'image_url', image_url: { url: imageDataUrl } }
                        ]
                    };
                }

                const openrouterBody = {
                    model: model,
                    messages: openrouterMessages
                };

                // Add web search tool if internet search is enabled
                if (isInternetSearch) {
                    openrouterBody.tools = [{ type: "openrouter:web_search" }];
                }

                const openrouterResponse = await fetch('https://openrouter.ai/api/v1/chat/completions', {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${openrouterApiKey}`,
                        'Content-Type': 'application/json',
                        'HTTP-Referer': req.headers.origin || 'http://localhost',
                        'X-Title': 'rene.gpt'
                    },
                    body: JSON.stringify(openrouterBody)
                });

                if (openrouterResponse.ok) {
                    const data = await openrouterResponse.json();
                    responseText = data.choices?.[0]?.message?.content || 'Inget svar från OpenRouter.';
                    providerUsed = 'openrouter';
                    modelUsed = model;
                    internetSearchUsed = isInternetSearch;
                } else {
                    const status = openrouterResponse.status;
                    let openRouterMessage = '';
                    try {
                        const errorData = await openrouterResponse.json();
                        openRouterMessage = errorData.error?.message || JSON.stringify(errorData);
                    } catch (e) {
                        openRouterMessage = openrouterResponse.statusText;
                    }

                    // Check if web search failed
                    if (isInternetSearch) {
                        const isWebSearchError = openRouterMessage.toLowerCase().includes('web_search') || 
                                                openRouterMessage.toLowerCase().includes('tool');
                        if (isWebSearchError) {
                            return res.status(status).json({ 
                                error: 'Internet search failed for this OpenRouter request.',
                                provider, 
                                model, 
                                status 
                            });
                        }
                    }

                    // Statuses eligible for fallback
                    const fallbackStatuses = [429, 500, 502, 503, 504];
                    if (fallbackStatuses.includes(status)) {
                        originalProviderError = `OpenRouter error: Status ${status}, Message: ${openRouterMessage}`;
                        // Proceed to fallback logic
                    } else {
                        // No fallback for 400, 401, 402, 404
                        let errorMsg = '';
                        switch (status) {
                            case 401:
                                errorMsg = "API key missing or invalid";
                                break;
                            case 402:
                                errorMsg = "OpenRouter credits/balance problem";
                                break;
                            case 404:
                                errorMsg = "Model ID not found or no endpoint available";
                                break;
                            case 400:
                                errorMsg = `OpenRouter message: ${openRouterMessage}`;
                                break;
                            default:
                                errorMsg = `HTTP ${status}: ${openRouterMessage}`;
                        }
                        return res.status(status).json({ error: errorMsg, provider, model, status });
                    }
                }
            } catch (error) {
                // Network error with OpenRouter, attempt fallback
                originalProviderError = `OpenRouter network error: ${error.message}`;
            }

            // If OpenRouter failed, attempt Gemini fallback
            if (!responseText) {
                if (!geminiApiKey) {
                    return res.status(400).json({ 
                        error: 'Missing GEMINI_API_KEY for fallback', 
                        originalProviderError 
                    });
                }

                // Fallback Gemini models (vision-capable, as per requirements)
                const fallbackGeminiModels = ['gemini-2.5-flash-lite', 'gemini-2.5-flash'];
                let geminiError = null;

                for (const geminiModel of fallbackGeminiModels) {
                    try {
                        const geminiContents = convertMessagesToGemini(messages, imageDataUrl, imageMimeType);
                        const geminiRequestBody = {
                            contents: geminiContents
                        };

                        // Add Google Search grounding if internet search is enabled
                        if (isInternetSearch) {
                            geminiRequestBody.tools = [{ googleSearch: {} }];
                        }

                        const geminiResponse = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${geminiModel}:generateContent`, {
                            method: 'POST',
                            headers: {
                                'x-goog-api-key': geminiApiKey,
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify(geminiRequestBody)
                        });

                        if (geminiResponse.ok) {
                            const geminiData = await geminiResponse.json();
                            responseText = geminiData.candidates?.[0]?.content?.parts?.[0]?.text || 'Inget svar från Gemini.';
                            providerUsed = 'gemini';
                            modelUsed = geminiModel;
                            fallbackUsed = true;
                            internetSearchUsed = isInternetSearch;

                            // Extract sources from grounding metadata if available
                            if (isInternetSearch && geminiData.candidates?.[0]?.groundingMetadata?.groundingChunks) {
                                sources = geminiData.candidates[0].groundingMetadata.groundingChunks
                                    .filter(chunk => chunk.web)
                                    .map(chunk => ({ 
                                        title: chunk.web.title || chunk.web.uri, 
                                        url: chunk.web.uri 
                                    }));
                            }
                            break;
                        } else {
                            const geminiStatus = geminiResponse.status;
                            let geminiMessage = '';
                            try {
                                const errorData = await geminiResponse.json();
                                geminiMessage = errorData.error?.message || JSON.stringify(errorData);
                            } catch (e) {
                                geminiMessage = geminiResponse.statusText;
                            }

                            // Check for Gemini internet search quota/availability errors
                            if (isInternetSearch) {
                                const isSearchUnavailable = geminiMessage.toLowerCase().includes('quota') || 
                                                          geminiMessage.toLowerCase().includes('grounding') ||
                                                          geminiMessage.toLowerCase().includes('unavailable');
                                if (isSearchUnavailable) {
                                    return res.status(geminiStatus).json({ 
                                        error: 'Gemini internet search is unavailable or quota-limited.',
                                        originalProviderError,
                                        geminiError: geminiMessage
                                    });
                                }
                            }

                            geminiError = `Gemini model ${geminiModel} failed: Status ${geminiStatus}, Message: ${geminiMessage}`;
                        }
                    } catch (error) {
                        geminiError = `Gemini model ${geminiModel} network error: ${error.message}`;
                    }
                }

                if (!responseText) {
                    return res.status(500).json({ 
                        error: 'Gemini fallback failed', 
                        originalProviderError, 
                        geminiError, 
                        fallbackUsed: true 
                    });
                }
            }
        } else if (provider === 'gemini') {
            // Direct Gemini request (chat)
            if (!geminiApiKey) {
                return res.status(400).json({ error: 'Missing GEMINI_API_KEY environment variable' });
            }

            const geminiContents = convertMessagesToGemini(messages, imageDataUrl, imageMimeType);
            const geminiRequestBody = {
                contents: geminiContents
            };

            // Add Google Search grounding if internet search is enabled
            if (isInternetSearch) {
                geminiRequestBody.tools = [{ googleSearch: {} }];
            }

            const geminiResponse = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
                method: 'POST',
                headers: {
                    'x-goog-api-key': geminiApiKey,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(geminiRequestBody)
            });

            if (geminiResponse.ok) {
                const geminiData = await geminiResponse.json();
                responseText = geminiData.candidates?.[0]?.content?.parts?.[0]?.text || 'Inget svar från Gemini.';
                providerUsed = 'gemini';
                modelUsed = model;
                internetSearchUsed = isInternetSearch;

                // Extract sources from grounding metadata if available
                if (isInternetSearch && geminiData.candidates?.[0]?.groundingMetadata?.groundingChunks) {
                    sources = geminiData.candidates[0].groundingMetadata.groundingChunks
                        .filter(chunk => chunk.web)
                        .map(chunk => ({ 
                            title: chunk.web.title || chunk.web.uri, 
                            url: chunk.web.uri 
                        }));
                }
            } else {
                const status = geminiResponse.status;
                let geminiMessage = '';
                try {
                    const errorData = await geminiResponse.json();
                    geminiMessage = errorData.error?.message || JSON.stringify(errorData);
                } catch (e) {
                    geminiMessage = geminiResponse.statusText;
                }

                // Check for Gemini internet search quota/availability errors
                if (isInternetSearch) {
                    const isSearchUnavailable = geminiMessage.toLowerCase().includes('quota') || 
                                              geminiMessage.toLowerCase().includes('grounding') ||
                                              geminiMessage.toLowerCase().includes('unavailable');
                    if (isSearchUnavailable) {
                        return res.status(status).json({ 
                            error: 'Gemini internet search is unavailable or quota-limited.',
                            provider, 
                            model, 
                            status 
                        });
                    }
                }

                return res.status(status).json({ 
                    error: `Gemini error: ${geminiMessage}`, 
                    provider, 
                    model, 
                    status 
                });
            }
        } else {
            return res.status(400).json({ error: `Unknown provider: ${provider}` });
        }

        // Return successful chat response
        return res.status(200).json({
            text: responseText,
            providerUsed,
            modelUsed,
            fallbackUsed,
            originalProviderError: originalProviderError || undefined,
            internetSearchUsed,
            sources: sources || null
        });

    } catch (error) {
        console.error('API /chat error:', error);
        return res.status(500).json({ error: `Internal server error: ${error.message}` });
    }
};

// Helper function to convert OpenAI-style messages to Gemini format
function convertMessagesToGemini(messages, imageDataUrl = null, imageMimeType = null) {
    const contents = [];
    let systemInstruction = '';

    // Extract system messages
    const systemMessages = messages.filter(msg => msg.role === 'system');
    if (systemMessages.length > 0) {
        systemInstruction = systemMessages.map(msg => msg.content).join('\n');
    }

    // Process non-system messages
    const nonSystemMessages = messages.filter(msg => msg.role !== 'system');
    for (let i = 0; i < nonSystemMessages.length; i++) {
        const msg = nonSystemMessages[i];
        if (msg.role === 'user') {
            let text = msg.content || '';
            // Prepend system instruction to first user message if exists
            if (contents.length === 0 && systemInstruction) {
                text = `${systemInstruction}\n\n${text}`;
            }
            const parts = [{ text: text }];

            // Add image to the last user message if imageDataUrl exists
            if (imageDataUrl && i === nonSystemMessages.length - 1) {
                // Strip data URL prefix (e.g., data:image/png;base64,)
                const base64Data = imageDataUrl.split(',')[1];
                if (!base64Data) {
                    throw new Error('Invalid image data URL: no base64 data found');
                }
                parts.push({
                    inline_data: {
                        mime_type: imageMimeType,
                        data: base64Data
                    }
                });
            }

            contents.push({
                role: 'user',
                parts: parts
            });
        } else if (msg.role === 'assistant') {
            contents.push({
                role: 'model',
                parts: [{ text: msg.content }]
            });
        }
    }

    return contents;
}

// Helper function to extract SEO data from HTML
function extractSeoData(html, pageUrl) {
    const data = {
        title: '',
        metaDescription: '',
        canonical: '',
        robots: '',
        h1: [],
        h2: [],
        h3: [],
        openGraph: {},
        twitterCard: {},
        jsonLd: [],
        imageCount: 0,
        imagesMissingAlt: 0,
        internalLinksCount: 0,
        externalLinksCount: 0,
        wordCount: 0
    };

    // Extract title
    const titleMatch = html.match(/<title>(.*?)<\/title>/i);
    if (titleMatch) data.title = titleMatch[1].trim();

    // Meta description
    const descMatch = html.match(/<meta\s+name="description"\s+content="(.*?)"/i) || html.match(/<meta\s+content="(.*?)"\s+name="description"/i);
    if (descMatch) data.metaDescription = descMatch[1].trim();

    // Canonical link
    const canonMatch = html.match(/<link\s+rel="canonical"\s+href="(.*?)"/i);
    if (canonMatch) data.canonical = canonMatch[1].trim();

    // Robots meta
    const robotsMatch = html.match(/<meta\s+name="robots"\s+content="(.*?)"/i);
    if (robotsMatch) data.robots = robotsMatch[1].trim();

    // H1 tags
    const h1Regex = /<h1[^>]*>(.*?)<\/h1>/gi;
    let h1Match;
    while ((h1Match = h1Regex.exec(html)) !== null) {
        data.h1.push(h1Match[1].replace(/<[^>]+>/g, '').trim());
    }

    // H2 tags
    const h2Regex = /<h2[^>]*>(.*?)<\/h2>/gi;
    let h2Match;
    while ((h2Match = h2Regex.exec(html)) !== null) {
        data.h2.push(h2Match[1].replace(/<[^>]+>/g, '').trim());
    }

    // H3 tags
    const h3Regex = /<h3[^>]*>(.*?)<\/h3>/gi;
    let h3Match;
    while ((h3Match = h3Regex.exec(html)) !== null) {
        data.h3.push(h3Match[1].replace(/<[^>]+>/g, '').trim());
    }

    // OpenGraph tags
    const ogRegex = /<meta\s+property="og:([^"]+)"\s+content="([^"]+)"/gi;
    let ogMatch;
    while ((ogMatch = ogRegex.exec(html)) !== null) {
        data.openGraph[ogMatch[1]] = ogMatch[2];
    }

    // Twitter card tags
    const twitterRegex = /<meta\s+name="twitter:([^"]+)"\s+content="([^"]+)"/gi;
    let twitterMatch;
    while ((twitterMatch = twitterRegex.exec(html)) !== null) {
        data.twitterCard[twitterMatch[1]] = twitterMatch[2];
    }

    // JSON-LD schema
    const jsonLdRegex = /<script\s+type="application\/ld\+json">(.*?)<\/script>/gis;
    let jsonLdMatch;
    while ((jsonLdMatch = jsonLdRegex.exec(html)) !== null) {
        try {
            const parsed = JSON.parse(jsonLdMatch[1]);
            data.jsonLd.push(parsed);
        } catch (e) {
            // Ignore invalid JSON
        }
    }

    // Image count and missing alt
    const imgRegex = /<img[^>]*>/gi;
    const imgMatches = html.match(imgRegex) || [];
    data.imageCount = imgMatches.length;
    imgMatches.forEach(img => {
        if (!/alt\s*=/i.test(img) || /alt=""/i.test(img)) {
            data.imagesMissingAlt++;
        }
    });

    // Links count
    const linkRegex = /<a[^>]*href="([^"]+)"[^>]*>/gi;
    let linkMatch;
    let pageHost = '';
    try {
        pageHost = new URL(pageUrl).hostname;
    } catch (e) {
        pageHost = '';
    }
    while ((linkMatch = linkRegex.exec(html)) !== null) {
        const href = linkMatch[1];
        if (href.startsWith('http')) {
            try {
                const linkHost = new URL(href).hostname;
                if (linkHost === pageHost) {
                    data.internalLinksCount++;
                } else {
                    data.externalLinksCount++;
                }
            } catch (e) {
                // ignore invalid URLs
            }
        } else if (!href.startsWith('#') && !href.startsWith('mailto:')) {
            // Relative links considered internal
            data.internalLinksCount++;
        }
    }

    // Word count (strip HTML tags)
    const textOnly = html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
    data.wordCount = textOnly.split(/\s+/).filter(w => w.length > 0).length;

    return data;
}

// Helper function to build analysis prompt based on type
function buildAnalysisPrompt(url, analysisType, seoData, userMessage) {
    let prompt = `You are an expert web analyst. Analyze the webpage at URL: ${url} for ${analysisType}.\n\n`;
    
    // Language detection instruction
    prompt += `LANGUAGE INSTRUCTION: `;
    if (userMessage) {
        prompt += `The user's message is: "${userMessage}". `;
    }
    prompt += `Detect the language of the user's message. If the message is mainly in Swedish, respond in Swedish. If it's mainly in English, respond in English. If there's no user message or it's unclear, default to Swedish.\n\n`;
    
    // Style instructions
    prompt += `OUTPUT STYLE INSTRUCTIONS:\n`;
    prompt += `- Use Markdown formatting with clear H2/H3 headings\n`;
    prompt += `- Add spacing between sections\n`;
    prompt += `- Use bullet lists for lists\n`;
    prompt += `- Avoid dense wall-of-text; use short paragraphs\n`;
    prompt += `- Keep tone professional, practical, and direct\n`;
    prompt += `- Do NOT start with a long disclaimer\n`;
    prompt += `- If the analysis is based only on parsed HTML (no screenshot), include a brief section called "Begränsning" (Swedish) or "Limitations" (English) with this text: "Analysen bygger på HTML, metadata, rubriker och länkar. Den bedömer inte den visuella layouten via screenshot." (Swedish) or "The analysis is based on HTML, metadata, headings, and links. It does not evaluate the visual layout via screenshot." (English)\n`;
    prompt += `- Do NOT translate website titles, URLs, brand names, headings, or quoted page text unless necessary\n`;
    prompt += `- Do NOT use English section titles like "Analysis Disclaimer" when responding in Swedish\n\n`;
    
    prompt += `Extracted SEO/Page Data:\n${JSON.stringify(seoData, null, 2)}\n\n`;
    
    // Structure based on language
    prompt += `RESPONSE STRUCTURE:\n\n`;
    
    prompt += `If responding in Swedish, use this structure:\n\n`;
    prompt += `## Snabb bedömning\n`;
    prompt += `[Ge en kort sammanfattning på 3-5 meningar]\n\n`;
    prompt += `## Det som fungerar bra\n`;
    prompt += `[Lista 3-5 styrkor]\n\n`;
    prompt += `## Problem jag ser\n`;
    prompt += `[Lista huvudproblemen, grupperade efter allvarlighetsgrad]\n\n`;
    prompt += `## Layout och användarupplevelse\n`;
    prompt += `[Ge praktisk feedback om layout. Om ingen skärmdump/rendrerad CSS finns tillgänglig, förklara tydligt att layoutfeedbacken härleds från HTML-strukturen.]\n\n`;
    prompt += `## SEO och struktur\n`;
    prompt += `[Diskutera title, meta description, H1/H2/H3, interna länkar, bilder, schema, canonical och innehållsstruktur om tillgängligt]\n\n`;
    prompt += `## Prioriterade förbättringar\n`;
    prompt += `[Ge 5-8 konkreta förbättringar. Varje rekommendation ska ha: vad som ska ändras, varför det spelar roll, hur man implementerar det kortfattat]\n\n`;
    prompt += `## Nästa bästa steg\n`;
    prompt += `[Ge 3 handlingssteg i prioritetsordning]\n\n`;
    
    prompt += `If responding in English, use this structure:\n\n`;
    prompt += `## Quick assessment\n`;
    prompt += `[Give a short 3-5 sentence summary]\n\n`;
    prompt += `## What works well\n`;
    prompt += `[List 3-5 strengths]\n\n`;
    prompt += `## Problems I see\n`;
    prompt += `[List the main issues, grouped by severity]\n\n`;
    prompt += `## Layout and user experience\n`;
    prompt += `[Give practical layout feedback. If no screenshot/rendered CSS is available, clearly say the layout feedback is inferred from HTML structure.]\n\n`;
    prompt += `## SEO and structure\n`;
    prompt += `[Discuss title, meta description, H1/H2/H3, internal links, images, schema, canonical, and content structure if available]\n\n`;
    prompt += `## Prioritized improvements\n`;
    prompt += `[Give 5-8 concrete improvements. Each recommendation should have: what to change, why it matters, how to implement it briefly]\n\n`;
    prompt += `## Next best steps\n`;
    prompt += `[Give 3 action steps in priority order]\n\n`;
    
    // Analysis type specific instructions
    switch (analysisType) {
        case 'SEO':
            prompt += `ANALYSIS TYPE: SEO\n`;
            prompt += `Focus more on metadata, headings, internal links, schema, content depth, and search intent.\n`;
            break;
        case 'Layout':
            prompt += `ANALYSIS TYPE: Layout\n`;
            prompt += `Include a dedicated section "Layoutförbättringar" (Swedish) or "Layout improvements" (English).\n`;
            prompt += `Always include concrete suggestions for:\n`;
            prompt += `- hero section\n- visual hierarchy\n- CTA buttons\n- mobile layout\n- spacing/section separation\n- navigation\n`;
            prompt += `If no screenshot is available, do not claim exact visual facts. Use wording like:\n`;
            prompt += `"Utifrån HTML-strukturen verkar..." (Swedish) or "Based on the HTML structure, it seems..." (English)\n`;
            prompt += `"Detta bör kontrolleras visuellt i webbläsaren..." (Swedish) or "This should be verified visually in the browser..." (English)\n`;
            break;
        case 'Copywriting':
            prompt += `ANALYSIS TYPE: Copywriting\n`;
            prompt += `Focus more on clarity, tone, CTA, trust, and conversion.\n`;
            break;
        case 'Accessibility':
            prompt += `ANALYSIS TYPE: Accessibility\n`;
            prompt += `Focus more on heading hierarchy, alt text, link labels, semantic structure, and mobile usability.\n`;
            break;
        case 'Technical':
            prompt += `ANALYSIS TYPE: Technical\n`;
            prompt += `Focus on technical aspects like HTML validation, meta tags, page speed factors, etc.\n`;
            break;
    }
    
    return prompt;
}
