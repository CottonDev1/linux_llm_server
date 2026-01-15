/**
 * Crypto Utilities for Password Encryption
 * Uses AES-256-GCM with Web Crypto API
 * For internal network use - provides obfuscation in localStorage
 */

const CryptoUtils = {
    // App-level secret (combined with user ID for key derivation)
    APP_SECRET: 'EWR-RAG-2024-Internal-Network-Key',

    /**
     * Derive an encryption key from user ID and app secret
     * Uses PBKDF2 for key derivation
     */
    async deriveKey(userId) {
        const encoder = new TextEncoder();
        const keyMaterial = await crypto.subtle.importKey(
            'raw',
            encoder.encode(this.APP_SECRET + userId),
            'PBKDF2',
            false,
            ['deriveKey']
        );

        return crypto.subtle.deriveKey(
            {
                name: 'PBKDF2',
                salt: encoder.encode('EWR-Salt-' + userId),
                iterations: 100000,
                hash: 'SHA-256'
            },
            keyMaterial,
            { name: 'AES-GCM', length: 256 },
            false,
            ['encrypt', 'decrypt']
        );
    },

    /**
     * Encrypt a string value
     * Returns base64 encoded string containing IV + ciphertext
     */
    async encrypt(plaintext, userId) {
        try {
            const key = await this.deriveKey(userId);
            const encoder = new TextEncoder();
            const iv = crypto.getRandomValues(new Uint8Array(12)); // 96-bit IV for GCM

            const ciphertext = await crypto.subtle.encrypt(
                { name: 'AES-GCM', iv: iv },
                key,
                encoder.encode(plaintext)
            );

            // Combine IV + ciphertext and encode as base64
            const combined = new Uint8Array(iv.length + ciphertext.byteLength);
            combined.set(iv);
            combined.set(new Uint8Array(ciphertext), iv.length);

            return btoa(String.fromCharCode(...combined));
        } catch (error) {
            console.error('Encryption failed:', error);
            return null;
        }
    },

    /**
     * Decrypt a base64 encoded string
     * Expects format: base64(IV + ciphertext)
     */
    async decrypt(encryptedBase64, userId) {
        try {
            const key = await this.deriveKey(userId);
            const decoder = new TextDecoder();

            // Decode base64
            const combined = Uint8Array.from(atob(encryptedBase64), c => c.charCodeAt(0));

            // Extract IV (first 12 bytes) and ciphertext
            const iv = combined.slice(0, 12);
            const ciphertext = combined.slice(12);

            const plaintext = await crypto.subtle.decrypt(
                { name: 'AES-GCM', iv: iv },
                key,
                ciphertext
            );

            return decoder.decode(plaintext);
        } catch (error) {
            console.error('Decryption failed:', error);
            return null;
        }
    },

    /**
     * Check if Web Crypto API is available
     */
    isSupported() {
        return !!(crypto && crypto.subtle);
    }
};

// Export for use in other scripts
if (typeof window !== 'undefined') {
    window.CryptoUtils = CryptoUtils;
}
