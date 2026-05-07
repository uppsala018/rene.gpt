module.exports = async (req, res) => {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    const { messages, provider, model } = req.body;

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

    try {
        if (provider === 'openrouter') {
            // Validate OpenRouter API key
            if (!openrouterApiKey) {
                return res.status(400).json({ error: 'Missing OPENROUTER_API_KEY environment variable' });
            }

            // Attempt OpenRouter request
            try {
                const openrouterResponse = await fetch('https://openrouter.ai/api/v1/chat/completions', {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${openrouterApiKey}`,
                        'Content-Type': 'application/json',
                        'HTTP-Referer': req.headers.origin || 'http://localhost',
                        'X-Title': 'rene.gpt'
                    },
                    body: JSON.stringify({
                        model: model,
                        messages: messages
                    })
                });

                if (openrouterResponse.ok) {
                    const data = await openrouterResponse.json();
                    responseText = data.choices?.[0]?.message?.content || 'Inget svar från OpenRouter.';
                    providerUsed = 'openrouter';
                    modelUsed = model;
                } else {
                    const status = openrouterResponse.status;
                    let openRouterMessage = '';
                    try {
                        const errorData = await openrouterResponse.json();
                        openRouterMessage = errorData.error?.message || JSON.stringify(errorData);
                    } catch (e) {
                        openRouterMessage = openrouterResponse.statusText;
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

                // Fallback Gemini models in order: gemini-2.5-flash-lite, then gemini-2.5-flash
                const fallbackGeminiModels = ['gemini-2.5-flash-lite', 'gemini-2.5-flash'];
                let geminiError = null;

                for (const geminiModel of fallbackGeminiModels) {
                    try {
                        const geminiContents = convertMessagesToGemini(messages);
                        const geminiResponse = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${geminiModel}:generateContent`, {
                            method: 'POST',
                            headers: {
                                'x-goog-api-key': geminiApiKey,
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                contents: geminiContents
                            })
                        });

                        if (geminiResponse.ok) {
                            const geminiData = await geminiResponse.json();
                            responseText = geminiData.candidates?.[0]?.content?.parts?.[0]?.text || 'Inget svar från Gemini.';
                            providerUsed = 'gemini';
                            modelUsed = geminiModel;
                            fallbackUsed = true;
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
            // Direct Gemini request
            if (!geminiApiKey) {
                return res.status(400).json({ error: 'Missing GEMINI_API_KEY environment variable' });
            }

            const geminiContents = convertMessagesToGemini(messages);
            const geminiResponse = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
                method: 'POST',
                headers: {
                    'x-goog-api-key': geminiApiKey,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    contents: geminiContents
                })
            });

            if (geminiResponse.ok) {
                const geminiData = await geminiResponse.json();
                responseText = geminiData.candidates?.[0]?.content?.parts?.[0]?.text || 'Inget svar från Gemini.';
                providerUsed = 'gemini';
                modelUsed = model;
            } else {
                const status = geminiResponse.status;
                let geminiMessage = '';
                try {
                    const errorData = await geminiResponse.json();
                    geminiMessage = errorData.error?.message || JSON.stringify(errorData);
                } catch (e) {
                    geminiMessage = geminiResponse.statusText;
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

        // Return successful response
        return res.status(200).json({
            text: responseText,
            providerUsed,
            modelUsed,
            fallbackUsed,
            originalProviderError: originalProviderError || undefined
        });

    } catch (error) {
        console.error('API /chat error:', error);
        return res.status(500).json({ error: `Internal server error: ${error.message}` });
    }
};

// Helper function to convert OpenAI-style messages to Gemini format
function convertMessagesToGemini(messages) {
    const contents = [];
    let systemInstruction = '';

    // Extract system messages
    const systemMessages = messages.filter(msg => msg.role === 'system');
    if (systemMessages.length > 0) {
        systemInstruction = systemMessages.map(msg => msg.content).join('\n');
    }

    // Process non-system messages
    const nonSystemMessages = messages.filter(msg => msg.role !== 'system');
    for (const msg of nonSystemMessages) {
        if (msg.role === 'user') {
            let text = msg.content;
            // Prepend system instruction to first user message if exists
            if (contents.length === 0 && systemInstruction) {
                text = `${systemInstruction}\n\n${text}`;
            }
            contents.push({
                role: 'user',
                parts: [{ text: text }]
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
