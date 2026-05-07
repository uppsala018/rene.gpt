module.exports = async (req, res) => {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    const { messages, provider, model, mode, prompt, imageDataUrl, imageMimeType, internetSearch } = req.body;
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
