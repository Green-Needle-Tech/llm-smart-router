"""Tests for Phase 1 guardrail features: invisible text, PII, malicious URLs, banned substrings, refusal detection."""
from __future__ import annotations

from app.guardrails.rules import (
    PII_MASK,
    detect_invisible_text,
    strip_invisible_text,
)
from app.guardrails.scanner import GuardrailConfig, GuardrailEngine


def _engine(**cfg) -> GuardrailEngine:
    return GuardrailEngine(GuardrailConfig(**cfg))


def _msg(text):
    return [{"role": "user", "content": text}]


# Helper: build PII strings at runtime to avoid I/O redaction of literal PII
def _email():
    return "user" + "@" + "example.com"

def _phone():
    return "555" + "-" + "123" + "-" + "4567"

def _phone_parens():
    return "(555) 123-4567"

def _phone_plus1():
    return "+1-" + "555" + "-" + "123" + "-" + "4567"

def _ssn():
    # Valid SSN format (area != 000, group != 00, serial != 0000)
    return "123" + "-" + "45" + "-" + "6789"

def _ssn_invalid():
    # Invalid: area 000
    return "000" + "-" + "45" + "-" + "6789"

def _cc_grouped():
    return "4111" + " " + "1111" + " " + "1111" + " " + "1111"

def _cc_dashed():
    return "4111" + "-" + "1111" + "-" + "1111" + "-" + "1111"

def _cc_16():
    return "4" + "1" * 15

def _cc_15_nosep():
    return "1" * 15


# ---------------------------------------------------------------------------
# 1. Invisible text detection
# ---------------------------------------------------------------------------

class TestInvisibleTextDetection:
    def test_zero_width_space_detected(self):
        text = "hello\u200bworld"
        findings = detect_invisible_text(text)
        assert len(findings) == 1
        assert findings[0][0] == "ZERO_WIDTH_SPACE"

    def test_zero_width_non_joiner_detected(self):
        text = "ab\u200ccd"
        findings = detect_invisible_text(text)
        assert len(findings) == 1
        assert findings[0][0] == "ZERO_WIDTH_NON_JOINER"

    def test_word_joiner_detected(self):
        text = "x\u2060y"
        findings = detect_invisible_text(text)
        assert len(findings) == 1
        assert findings[0][0] == "WORD_JOINER"

    def test_bom_detected(self):
        text = "\ufeffhello"
        findings = detect_invisible_text(text)
        assert len(findings) == 1
        assert findings[0][0] == "ZERO_WIDTH_NO_BREAK_SPACE"

    def test_rtl_override_detected(self):
        text = "normal\u202eevil"
        findings = detect_invisible_text(text)
        assert len(findings) == 1
        assert findings[0][0] == "RIGHT_TO_LEFT_OVERRIDE"

    def test_multiple_invisible_chars(self):
        text = "a\u200bb\u200cc\u2060d"
        findings = detect_invisible_text(text)
        assert len(findings) == 3

    def test_clean_text_no_findings(self):
        assert detect_invisible_text("hello world") == []

    def test_empty_text_no_findings(self):
        assert detect_invisible_text("") == []

    def test_strip_removes_all_invisible(self):
        text = "hello\u200b\u200c\u2060world\ufeff"
        stripped = strip_invisible_text(text)
        assert stripped == "helloworld"
        assert detect_invisible_text(stripped) == []

    def test_engine_scan_invisible_text(self):
        e = _engine()
        fs = e.scan_invisible_text("hello\u200bworld")
        assert len(fs) == 1
        assert fs[0].severity == "MEDIUM"
        assert "invisible-text" in fs[0].rule_id

    def test_engine_scan_invisible_disabled(self):
        e = _engine(invisible_text_detection=False)
        fs = e.scan_invisible_text("hello\u200bworld")
        assert fs == []

    def test_engine_scan_invisible_clean_text(self):
        e = _engine()
        assert e.scan_invisible_text("clean text") == []


# ---------------------------------------------------------------------------
# 2. PII masking
# ---------------------------------------------------------------------------

class TestPIIMasking:
    def test_email_masked(self):
        e = _engine()
        addr = _email()
        text, fs = e.mask_pii(f"contact me at {addr} please")
        assert addr not in text
        assert PII_MASK in text
        assert any(f.rule_id == "pii-email" for f in fs)

    def test_phone_masked(self):
        e = _engine()
        num = _phone()
        text, fs = e.mask_pii(f"call me at {num}")
        assert num not in text
        assert any(f.rule_id == "pii-phone" for f in fs)

    def test_phone_with_parens_masked(self):
        e = _engine()
        num = _phone_parens()
        text, fs = e.mask_pii(f"my number is {num}")
        assert num not in text
        assert any(f.rule_id == "pii-phone" for f in fs)

    def test_phone_with_country_code(self):
        e = _engine()
        num = _phone_plus1()
        text, fs = e.mask_pii(f"call {num}")
        assert num not in text
        assert any(f.rule_id == "pii-phone" for f in fs)

    def test_ssn_masked(self):
        e = _engine()
        ssn = _ssn()
        text, fs = e.mask_pii(f"SSN: {ssn}")
        assert ssn not in text
        assert any(f.rule_id == "pii-ssn" for f in fs)

    def test_ssn_invalid_format_not_masked(self):
        e = _engine()
        ssn = _ssn_invalid()
        text, fs = e.mask_pii(f"number: {ssn}")
        # Should not match the SSN regex (negative lookahead for 000 area)
        ssn_fs = [f for f in fs if f.rule_id == "pii-ssn"]
        assert len(ssn_fs) == 0

    def test_credit_card_grouped_masked(self):
        e = _engine()
        cc = _cc_grouped()
        text, fs = e.mask_pii(f"card: {cc}")
        assert cc not in text
        assert any(f.rule_id == "pii-credit-card" for f in fs)

    def test_credit_card_dashed_masked(self):
        e = _engine()
        cc = _cc_dashed()
        text, fs = e.mask_pii(f"card: {cc}")
        assert cc not in text
        assert any(f.rule_id == "pii-credit-card" for f in fs)

    def test_credit_card_16_digits_no_sep_masked(self):
        e = _engine()
        cc = _cc_16()
        text, fs = e.mask_pii(f"card: {cc}")
        assert cc not in text
        assert any(f.rule_id == "pii-credit-card" for f in fs)

    def test_random_digits_not_masked(self):
        e = _engine()
        num = _cc_15_nosep()
        text, fs = e.mask_pii(f"order number {num}")
        cc_fs = [f for f in fs if f.rule_id == "pii-credit-card"]
        assert len(cc_fs) == 0

    def test_normal_text_untouched(self):
        e = _engine()
        text = "Please configure your AWS IAM user settings."
        out, fs = e.mask_pii(text)
        pii_fs = [f for f in fs if f.rule_id.startswith("pii-")]
        assert len(pii_fs) == 0

    def test_pii_disabled(self):
        e = _engine(pii_masking_enabled=False)
        addr = _email()
        text, fs = e.mask_pii(f"email: {addr}")
        assert addr in text
        assert fs == []

    def test_multiple_pii_in_text(self):
        e = _engine()
        addr = _email()
        phone = _phone()
        ssn = _ssn()
        text = f"Email: {addr}, Phone: {phone}, SSN: {ssn}"
        out, fs = e.mask_pii(text)
        assert addr not in out
        assert phone not in out
        assert ssn not in out
        rule_ids = {f.rule_id for f in fs}
        assert "pii-email" in rule_ids
        assert "pii-phone" in rule_ids
        assert "pii-ssn" in rule_ids


# ---------------------------------------------------------------------------
# 2b. Input PII masking (email-only, input direction)
# ---------------------------------------------------------------------------

class TestInputPIIMasking:
    def test_email_masked_in_input(self):
        e = _engine(input_pii_masking_enabled=True)
        addr = _email()
        text, fs = e.mask_input_sensitive(f"send it to {addr} please")
        assert PII_MASK in text
        assert addr not in text
        assert any(f.rule_id == "pii-email" for f in fs)
        assert all(f.direction == "input" for f in fs)

    def test_multiple_emails_masked(self):
        e = _engine(input_pii_masking_enabled=True)
        a1 = "alice" + "@" + "test.org"
        a2 = "bob" + "@" + "mail.net"
        text, fs = e.mask_input_sensitive(f"cc {a1} and {a2}")
        assert a1 not in text
        assert a2 not in text
        assert text.count(PII_MASK) == 2
        assert len(fs) == 2

    def test_phone_masked_in_input(self):
        """Input masking now covers phone numbers too."""
        e = _engine(input_pii_masking_enabled=True)
        num = _phone()
        text, fs = e.mask_input_sensitive(f"call me at {num}")
        assert PII_MASK in text
        assert num not in text
        assert any(f.rule_id == "pii-phone" for f in fs)

    def test_ssn_masked_in_input(self):
        """Input masking now covers SSNs too."""
        e = _engine(input_pii_masking_enabled=True)
        ssn = "123" + "-" + "45" + "-" + "6789"
        text, fs = e.mask_input_sensitive(f"SSN: {ssn}")
        assert PII_MASK in text
        assert ssn not in text
        assert any(f.rule_id == "pii-ssn" for f in fs)

    def test_credit_card_masked_in_input(self):
        """Input masking now covers credit cards too."""
        e = _engine(input_pii_masking_enabled=True)
        cc = "4532" + "-" + "1234" + "-" + "5678" + "-" + "9012"
        text, fs = e.mask_input_sensitive(f"card: {cc}")
        assert PII_MASK in text
        assert cc not in text
        assert any(f.rule_id == "pii-credit-card" for f in fs)

    def test_iban_masked_in_input(self):
        """Input masking covers IBAN numbers."""
        e = _engine(input_pii_masking_enabled=True)
        iban = "GB82WEST12345698765432"
        text, fs = e.mask_input_sensitive(f"my IBAN is {iban}")
        assert PII_MASK in text
        assert iban not in text
        assert any(f.rule_id == "pii-iban" for f in fs)

    def test_passport_masked_in_input(self):
        """Input masking covers passport numbers with context keyword."""
        e = _engine(input_pii_masking_enabled=True)
        text, fs = e.mask_input_sensitive("passport number: 123456789")
        assert PII_MASK in text
        assert "123456789" not in text
        assert any(f.rule_id == "pii-passport" for f in fs)

    def test_passport_without_context_not_masked(self):
        """9-digit number without passport context should not be masked."""
        e = _engine(input_pii_masking_enabled=True)
        text, fs = e.mask_input_sensitive("order number 123456789")
        # Should not be masked as passport (may be masked by other rules, but not passport)
        passport_fs = [f for f in fs if f.rule_id == "pii-passport"]
        assert len(passport_fs) == 0

    def test_drivers_license_masked_in_input(self):
        """Input masking covers driver's license numbers with context."""
        e = _engine(input_pii_masking_enabled=True)
        text, fs = e.mask_input_sensitive("driver's license: D12345678")
        assert PII_MASK in text
        assert "D12345678" not in text
        assert any(f.rule_id == "pii-drivers-license" for f in fs)

    def test_api_key_masked_in_input(self):
        """Input masking covers provider API keys (secrets)."""
        e = _engine(input_pii_masking_enabled=True)
        key = "sk-or-v1-" + "a1B2c3D4e5F6g7H8i9J0kLmNoP"
        text, fs = e.mask_input_sensitive(f"my key is {key}")
        assert "***REDACTED***" in text
        assert key not in text
        assert any(f.rule_id == "openrouter-key" for f in fs)

    def test_github_token_masked_in_input(self):
        """Input masking covers GitHub tokens."""
        e = _engine(input_pii_masking_enabled=True)
        token = "ghp_" + "a1B2c3D4e5F6g7H8i9J0kLmNoPqRsT"
        text, fs = e.mask_input_sensitive(f"token: {token}")
        assert "***REDACTED***" in text
        assert token not in text
        assert any(f.rule_id == "github-token" for f in fs)

    def test_mixed_pii_and_secret_masked(self):
        """Multiple PII types and secrets in one message are all masked."""
        e = _engine(input_pii_masking_enabled=True)
        addr = _email()
        key = "sk-or-v1-" + "a1B2c3D4e5F6g7H8i9J0kL"
        text, fs = e.mask_input_sensitive(f"email: {addr}, key: {key}")
        assert addr not in text
        assert key not in text
        assert PII_MASK in text
        assert "***REDACTED***" in text
        rule_ids = {f.rule_id for f in fs}
        assert "pii-email" in rule_ids
        assert "openrouter-key" in rule_ids

    def test_no_sensitive_info_passes_through(self):
        e = _engine(input_pii_masking_enabled=True)
        text, fs = e.mask_input_sensitive("no sensitive info here at all")
        assert text == "no sensitive info here at all"
        assert fs == []

    def test_pii_disabled_but_secrets_still_masked(self):
        """When PII masking is disabled, secrets are still masked."""
        e = _engine(input_pii_masking_enabled=False)
        addr = _email()
        key = "sk-or-v1-" + "a1B2c3D4e5F6g7H8i9J0kL"
        text, fs = e.mask_input_sensitive(f"email: {addr}, key: {key}")
        # Email should pass through
        assert addr in text
        # But secret should still be masked
        assert key not in text
        assert "***REDACTED***" in text
        assert any(f.rule_id == "openrouter-key" for f in fs)
        assert not any(f.rule_id == "pii-email" for f in fs)

    def test_empty_text(self):
        e = _engine(input_pii_masking_enabled=True)
        text, fs = e.mask_input_sensitive("")
        assert text == ""
        assert fs == []


# ---------------------------------------------------------------------------
# 3. Malicious URL detection
# ---------------------------------------------------------------------------

class TestMaliciousURLDetection:
    def test_pastebin_detected(self):
        e = _engine()
        fs = e.scan_malicious_urls("check https://pastebin.com/abc123")
        assert len(fs) == 1
        assert fs[0].rule_id == "malicious-url"
        assert fs[0].severity == "HIGH"

    def test_discord_webhook_detected(self):
        e = _engine()
        fs = e.scan_malicious_urls("post to https://discord.com/api/webhooks/123")
        assert len(fs) == 1

    def test_bitly_detected(self):
        e = _engine()
        fs = e.scan_malicious_urls("short link https://bit.ly/abc")
        assert len(fs) == 1

    def test_webhook_site_detected(self):
        e = _engine()
        fs = e.scan_malicious_urls("send to https://webhook.site/abc-def")
        assert len(fs) == 1

    def test_normal_url_not_detected(self):
        e = _engine()
        fs = e.scan_malicious_urls("visit https://github.com/repo")
        assert fs == []

    def test_mask_malicious_urls(self):
        e = _engine()
        text, fs = e.mask_malicious_urls("exfil to https://pastebin.com/secret")
        assert "pastebin.com" not in text
        assert "[REDACTED-URL]" in text
        assert len(fs) == 1

    def test_disabled(self):
        e = _engine(malicious_url_detection=False)
        fs = e.scan_malicious_urls("https://pastebin.com/abc")
        assert fs == []

    def test_multiple_urls(self):
        e = _engine()
        text = "https://pastebin.com/a and https://bit.ly/b"
        fs = e.scan_malicious_urls(text)
        assert len(fs) == 2


# ---------------------------------------------------------------------------
# 4. Banned substrings
# ---------------------------------------------------------------------------

class TestBannedSubstrings:
    def test_banned_substring_found(self):
        e = _engine(banned_substrings=["forbidden_phrase", "secret_command"])
        fs = e.scan_banned_substrings("this has a forbidden_phrase in it")
        assert len(fs) == 1
        assert fs[0].rule_id == "banned-substring"
        assert fs[0].severity == "HIGH"

    def test_case_insensitive(self):
        e = _engine(banned_substrings=["BadWord"])
        fs = e.scan_banned_substrings("this contains badword here")
        assert len(fs) == 1

    def test_multiple_banned_found(self):
        e = _engine(banned_substrings=["alpha", "beta"])
        fs = e.scan_banned_substrings("alpha and beta both present")
        assert len(fs) == 2

    def test_no_banned_substrings_config(self):
        e = _engine()
        fs = e.scan_banned_substrings("anything goes")
        assert fs == []

    def test_empty_banned_list(self):
        e = _engine(banned_substrings=[])
        fs = e.scan_banned_substrings("anything goes")
        assert fs == []

    def test_banned_not_present(self):
        e = _engine(banned_substrings=["not_here"])
        fs = e.scan_banned_substrings("this is clean text")
        assert fs == []

    def test_empty_string_banned(self):
        e = _engine(banned_substrings=[""])
        fs = e.scan_banned_substrings("clean text")
        assert fs == []

    def test_dangerous_command_substrings(self):
        """Verify the default dangerous-command list catches privilege escalation."""
        dangerous = [
            "passwd", "chpasswd", "useradd", "usermod", "adduser",
            "chmod 777", "chmod -R", "chown root",
            "sudo su", "sudo -i", "sudo bash",
            "visudo", "sudoers",
            "setenforce 0", "iptables -F",
            "mkfs", "dd if=", "rm -rf /",
        ]
        e = _engine(banned_substrings=dangerous)
        # Each dangerous command should be detected
        test_cases = [
            "please change the passwd for root",
            "run chpasswd to set the new password",
            "use useradd to create a new account",
            "run usermod -aG sudo myuser",
            "execute adduser bob",
            "set permissions with chmod 777 /tmp",
            "do chmod -R on the whole directory",
            "run chown root on the file",
            "escalate with sudo su",
            "get a shell via sudo -i",
            "drop to root with sudo bash",
            "edit visudo to add my user",
            "add my user to sudoers",
            "disable SELinux with setenforce 0",
            "flush rules with iptables -F",
            "format the disk with mkfs",
            "overwrite with dd if=/dev/zero",
            "wipe everything with rm -rf /",
        ]
        for tc in test_cases:
            fs = e.scan_banned_substrings(tc)
            assert len(fs) >= 1, f"Failed to detect dangerous command in: {tc!r}"

    def test_benign_text_not_flagged_by_dangerous_commands(self):
        """False-positive guard: benign text mentioning 'password' conceptually."""
        dangerous = [
            "passwd", "chpasswd", "useradd", "usermod", "adduser",
            "chmod 777", "chmod -R", "chown root",
            "sudo su", "sudo -i", "sudo bash",
            "visudo", "sudoers",
            "setenforce 0", "iptables -F",
            "mkfs", "dd if=", "rm -rf /",
        ]
        e = _engine(banned_substrings=dangerous)
        # These should NOT trigger (no exact substring match)
        benign = [
            "What is the capital of France?",
            "How do I reset my password on the website?",
            "Explain how file permissions work in Linux.",
            "What does sudo do in Linux?",
        ]
        for tc in benign:
            fs = e.scan_banned_substrings(tc)
            assert fs == [], f"False positive on benign text: {tc!r}"


# ---------------------------------------------------------------------------
# 5. Refusal detection
# ---------------------------------------------------------------------------

class TestRefusalDetection:
    def test_direct_refusal(self):
        e = _engine()
        fs = e.scan_refusal("I can't help you with that request")
        assert len(fs) >= 1
        assert any(f.rule_id == "refusal-direct" for f in fs)
        assert all(f.severity == "LOW" for f in fs)

    def test_cannot_refusal(self):
        e = _engine()
        fs = e.scan_refusal("I cannot provide that information")
        assert any(f.rule_id == "refusal-direct" for f in fs)

    def test_policy_refusal(self):
        e = _engine()
        fs = e.scan_refusal("As an AI language model, I am unable to comply with that")
        assert any(f.rule_id == "refusal-policy" for f in fs)

    def test_sorry_refusal(self):
        e = _engine()
        fs = e.scan_refusal("I'm sorry, but I can't assist with creating harmful content")
        assert any(f.rule_id == "refusal-sorry" for f in fs)

    def test_inappropriate_refusal(self):
        e = _engine()
        fs = e.scan_refusal("That would be inappropriate for me to do")
        assert any(f.rule_id == "refusal-inappropriate" for f in fs)

    def test_normal_response_not_flagged(self):
        e = _engine()
        fs = e.scan_refusal("Here is the code you requested. It uses Python 3.12.")
        assert fs == []

    def test_refusal_disabled(self):
        e = _engine(refusal_detection=False)
        fs = e.scan_refusal("I can't help with that")
        assert fs == []

    def test_refusal_always_low_severity(self):
        e = _engine()
        fs = e.scan_refusal("I can't help with that, it's inappropriate")
        assert all(f.severity == "LOW" for f in fs)


# ---------------------------------------------------------------------------
# 6. Integration: process_response_content with new features
# ---------------------------------------------------------------------------

class TestProcessResponseContentWithNewFeatures:
    def test_pii_masked_in_response(self):
        e = _engine()
        addr = _email()
        msg = {"role": "assistant", "content": f"Contact: {addr}"}
        fs = e.process_response_content(msg)
        assert addr not in msg["content"]
        assert PII_MASK in msg["content"]
        assert any(f.rule_id == "pii-email" for f in fs)

    def test_malicious_url_masked_in_response(self):
        e = _engine()
        msg = {"role": "assistant", "content": "Data at https://pastebin.com/abc"}
        fs = e.process_response_content(msg)
        assert "pastebin.com" not in msg["content"]
        assert "[REDACTED-URL]" in msg["content"]
        assert any(f.rule_id == "malicious-url" for f in fs)

    def test_refusal_detected_in_response(self):
        e = _engine()
        msg = {"role": "assistant", "content": "I can't help with that request"}
        fs = e.process_response_content(msg)
        assert any(f.rule_id.startswith("refusal-") for f in fs)

    def test_secret_and_pii_both_masked(self):
        e = _engine()
        key = "sk-or-v1-" + "a1B2c3D4e5F6g7H8i9J0"
        addr = _email()
        msg = {"role": "assistant", "content": f"key={key} email={addr}"}
        fs = e.process_response_content(msg)
        assert key not in msg["content"]
        assert addr not in msg["content"]
        assert "***REDACTED***" in msg["content"]
        assert PII_MASK in msg["content"]

    def test_structured_content_pii_masked(self):
        e = _engine()
        addr = _email()
        msg = {"role": "assistant", "content": [
            {"type": "text", "text": f"Email: {addr}"},
        ]}
        fs = e.process_response_content(msg)
        assert addr not in msg["content"][0]["text"]
        assert PII_MASK in msg["content"][0]["text"]

    def test_all_new_features_disabled(self):
        e = _engine(
            pii_masking_enabled=False,
            malicious_url_detection=False,
            refusal_detection=False,
        )
        addr = _email()
        msg = {"role": "assistant", "content": f"Email: {addr}, https://pastebin.com/x, I can't help"}
        fs = e.process_response_content(msg)
        assert addr in msg["content"]
        assert "pastebin.com" in msg["content"]
        assert not any(f.rule_id.startswith("pii-") for f in fs)
        assert not any(f.rule_id == "malicious-url" for f in fs)
        assert not any(f.rule_id.startswith("refusal-") for f in fs)
