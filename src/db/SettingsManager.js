/**
 * Settings Manager
 * Centralized settings management for system configuration
 */

class SettingsManager {
  constructor(ewraiDatabase) {
    this.db = ewraiDatabase;
  }

  /**
   * Get a single setting value
   * @param {string} section - Setting key (e.g., 'AuthMode')
   * @returns {string|null} Setting value or null if not found
   */
  getSetting(section) {
    // Bypass validation - return null if DB not initialized
    if (!this.db || !this.db.db) {
      return null;
    }
    try {
      return this.db.getSetting(section);
    } catch (error) {
      console.warn(`SettingsManager: Failed to get setting ${section}:`, error.message);
      return null;
    }
  }

  /**
   * Get all settings as an object
   * @returns {Object} All settings as key-value pairs
   */
  getAllSettings() {
    // Bypass validation - return empty object if DB not initialized
    if (!this.db || !this.db.db) {
      return {};
    }
    try {
      return this.db.getAllSettings();
    } catch (error) {
      console.warn('SettingsManager: Failed to get all settings:', error.message);
      return {};
    }
  }

  /**
   * Update a single setting
   * @param {string} section - Setting key
   * @param {string} value - Setting value
   */
  updateSetting(section, value) {
    this.db.updateSetting(section, value);
  }

  /**
   * Update multiple settings at once
   * @param {Object} settings - Object with section-value pairs
   */
  updateSettings(settings) {
    this.db.updateSystemSettings(settings);
  }

  /**
   * Delete a setting (sets it to empty string)
   * @param {string} section - Setting key to delete
   */
  deleteSetting(section) {
    this.db.updateSetting(section, '');
  }

  /**
   * Get authentication mode
   * @returns {string} 'SQL' or 'Windows'
   */
  getAuthMode() {
    return this.getSetting('AuthMode') || 'SQL';
  }

  /**
   * Set authentication mode
   * @param {string} mode - 'SQL' or 'Windows'
   */
  setAuthMode(mode) {
    if (mode !== 'SQL' && mode !== 'Windows') {
      throw new Error('Invalid auth mode. Must be "SQL" or "Windows"');
    }
    this.updateSetting('AuthMode', mode);
  }

  /**
   * Get Windows domain settings
   * @returns {Object} Windows auth configuration
   */
  getWindowsAuthSettings() {
    return {
      domain: this.getSetting('WindowsDomain') || '',
      domainController: this.getSetting('WindowsDC') || '',
      ldapBaseDN: this.getSetting('LDAPBaseDN') || '',
      ldapBindUser: this.getSetting('LDAPBindUser') || '',
      ldapBindPassword: this.getSetting('LDAPBindPassword') || '',
      adminGroups: (this.getSetting('ADAdminGroups') || 'Domain Admins,IT Admins').split(',').map(g => g.trim()),
      developerGroups: (this.getSetting('ADDeveloperGroups') || 'Developers,Engineering').split(',').map(g => g.trim())
    };
  }

  /**
   * Update Windows domain settings
   * @param {Object} settings - Windows auth configuration
   */
  updateWindowsAuthSettings(settings) {
    const updates = {};

    if (settings.domain !== undefined) {
      updates.WindowsDomain = settings.domain;
    }
    if (settings.domainController !== undefined) {
      updates.WindowsDC = settings.domainController;
    }
    if (settings.ldapBaseDN !== undefined) {
      updates.LDAPBaseDN = settings.ldapBaseDN;
    }
    if (settings.ldapBindUser !== undefined) {
      updates.LDAPBindUser = settings.ldapBindUser;
    }
    if (settings.ldapBindPassword !== undefined) {
      updates.LDAPBindPassword = settings.ldapBindPassword;
    }
    if (settings.adminGroups !== undefined) {
      updates.ADAdminGroups = Array.isArray(settings.adminGroups)
        ? settings.adminGroups.join(',')
        : settings.adminGroups;
    }
    if (settings.developerGroups !== undefined) {
      updates.ADDeveloperGroups = Array.isArray(settings.developerGroups)
        ? settings.developerGroups.join(',')
        : settings.developerGroups;
    }

    this.updateSettings(updates);
  }

  /**
   * Check if Windows authentication is enabled
   * @returns {boolean}
   */
  isWindowsAuthEnabled() {
    return this.getAuthMode() === 'Windows';
  }

  /**
   * Validate Windows auth settings are configured
   * @returns {Object} { valid: boolean, missingFields: string[] }
   */
  validateWindowsAuthSettings() {
    const settings = this.getWindowsAuthSettings();
    const requiredFields = ['domain', 'domainController', 'ldapBaseDN'];
    const missingFields = [];

    for (const field of requiredFields) {
      if (!settings[field] || settings[field].trim() === '') {
        missingFields.push(field);
      }
    }

    return {
      valid: missingFields.length === 0,
      missingFields
    };
  }

  // ========================================================================
  // AUDIO SETTINGS
  // ========================================================================

  /**
   * Get audio summary length threshold in seconds
   * When audio duration exceeds this, only summary is generated (not full transcription)
   * @returns {number} Threshold in seconds (0 = disabled, always full transcription)
   */
  getAudioSummaryThreshold() {
    const value = this.getSetting('AudioSummaryThreshold');
    return value ? parseInt(value, 10) : 0;
  }

  /**
   * Set audio summary length threshold
   * @param {number} seconds - Threshold in seconds (0 to disable)
   */
  setAudioSummaryThreshold(seconds) {
    const value = Math.max(0, parseInt(seconds, 10) || 0);
    this.updateSetting('AudioSummaryThreshold', value.toString());
    return value;
  }

  /**
   * Get audio summary threshold enabled state
   * @returns {boolean} True if threshold is enabled (> 0)
   */
  isAudioSummaryThresholdEnabled() {
    return this.getAudioSummaryThreshold() > 0;
  }

  /**
   * Get all audio-related settings
   * @returns {Object} Audio settings object
   */
  getAudioSettings() {
    return {
      summaryThreshold: this.getAudioSummaryThreshold(),
      summaryThresholdEnabled: this.isAudioSummaryThresholdEnabled(),
      bulkFolderPath: this.getSetting('AudioBulkFolderPath') || ''
    };
  }

  /**
   * Update multiple audio settings at once
   * @param {Object} settings - Audio settings object
   */
  updateAudioSettings(settings) {
    const updates = {};

    if (settings.summaryThreshold !== undefined) {
      updates.AudioSummaryThreshold = Math.max(0, parseInt(settings.summaryThreshold, 10) || 0).toString();
    }

    if (settings.bulkFolderPath !== undefined) {
      updates.AudioBulkFolderPath = settings.bulkFolderPath;
    }

    if (Object.keys(updates).length > 0) {
      this.updateSettings(updates);
    }

    return this.getAudioSettings();
  }
}

export default SettingsManager;
