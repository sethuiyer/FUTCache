"""Shared data + model helper for the FUTCache demos.

Centralizes the canonical 45-sentence support corpus, the two question sets
(monolingual 10 intents x 3 phrasings; cross-lingual 13 unique intents + non-
English re-asks), and a cached embedding-model loader, so every demo reads the
same data. Import with ``_shared`` (the demos live in ``demos/`` and are run as
``python demos/<name>.py``, which puts ``demos/`` on ``sys.path``).
"""
import numpy as np

MODEL = "hotchpotch/bekko-embedding-v1-a8m"

CORPUS = [
    "You can reset your password from the account settings page.",
    "Password resets require a confirmation link sent to your email.",
    "Choose a password with at least eight characters and a number.",
    "A password reset link expires after sixty minutes.",
    "Contact support if the reset email does not arrive.",
    "To sign out, open the user menu and select logout.",
    "Logging out clears your local session on the device.",
    "You stay signed in on trusted devices by default.",
    "Ending a session on a shared computer is recommended.",
    "Subscriptions are managed in the billing section.",
    "You can cancel a subscription at the end of the billing cycle.",
    "Cancelling stops renewals but keeps access until the period ends.",
    "A confirmation email is sent when a subscription is cancelled.",
    "Downgrades take effect on the next billing date.",
    "Refunds are issued within five to ten business days.",
    "Eligible refunds are processed back to your original payment method.",
    "Contact billing to request a refund for a recent charge.",
    "Pro-rated refunds apply to annual plans.",
    "API keys are generated under the developer settings tab.",
    "Keep your API key secret; do not commit it to a repository.",
    "You can create multiple API keys for different environments.",
    "Rotate an API key if you suspect it has leaked.",
    "Change your email address from the profile settings.",
    "A verification email is sent to the new address.",
    "Updating your email does not affect your account data.",
    "Invoices are available for download in the billing history.",
    "Each invoice includes a line-by-line summary of charges.",
    "Receipts are emailed at the end of each billing cycle.",
    "A mobile app is available for iOS and Android.",
    "The mobile app supports the same features as the web version.",
    "Push notifications are configurable in the mobile settings.",
    "There is a free tier with a monthly usage limit.",
    "Paid plans unlock higher usage caps and priority support.",
    "You are billed monthly, with an option to pay annually.",
    "Two-factor authentication adds a second step at sign in.",
    "Enable 2FA from the security settings on your account.",
    "Authenticator apps generate the verification code.",
    "Backup codes let you sign in if your device is lost.",
    "You can invite teammates from the organization settings.",
    "Team roles control who can change billing details.",
    "Audit logs record changes made to your workspace.",
    "Data is encrypted at rest and in transit.",
    "You can export your data in a CSV file.",
    "The help center has step-by-step guides for common tasks.",
    "Session length and timeout are configurable by admins.",
]

QUESTIONS_MONO = {
    "password_reset": [
        "How do I reset my password?",
        "What is the procedure to change my password?",
        "I forgot my password, how can I get a new one?",
    ],
    "sign_out": [
        "How do I log out of my account?",
        "What is the way to sign out?",
        "How can I end my current session?",
    ],
    "cancel_subscription": [
        "How do I cancel my subscription?",
        "What is the process to cancel my plan?",
        "I want to stop my subscription, how?",
    ],
    "refund": [
        "How do I get a refund?",
        "What is the refund process?",
        "Can I get my money back, and how?",
    ],
    "api_key": [
        "Where do I find my API key?",
        "How do I generate an API key?",
        "How do I get my API credentials?",
    ],
    "change_email": [
        "How do I change my email address?",
        "What is the way to update my email?",
        "How can I modify the email on my account?",
    ],
    "invoice": [
        "Where can I download my invoice?",
        "How do I get a copy of my bill?",
        "How do I access my billing statement?",
    ],
    "mobile_app": [
        "Is there a mobile app?",
        "Do you have an iOS or Android app?",
        "Can I use this from my phone?",
    ],
    "pricing": [
        "How much does it cost?",
        "What are the pricing tiers?",
        "What is the monthly price?",
    ],
    "two_factor": [
        "How do I enable 2FA?",
        "What is the process to turn on two-factor authentication?",
        "How do I set up multifactor authentication?",
    ],
}

QUESTIONS_MULTI = {
    "reset_password": [
        ("en", "How do I reset my password?"),
        ("es", "¿Cómo restablezco mi contraseña?"),
        ("fr", "Comment réinitialiser mon mot de passe ?"),
        ("ja", "パスワードをリセットするにはどうすればよいですか？"),
    ],
    "cancel_subscription": [
        ("en", "How do I cancel my subscription?"),
        ("de", "Wie kündige ich mein Abo?"),
        ("zh", "我如何取消我的订阅？"),
    ],
    "refund": [
        ("en", "How do I get a refund?"),
        ("es", "¿Cómo puedo solicitar un reembolso?"),
        ("pt", "Como faço para obter um reembolso?"),
        ("de", "Wie bekomme ich eine Rückerstattung?"),
    ],
    "contact_support": [
        ("en", "How do I contact support?"),
        ("fr", "Comment contacter le support ?"),
        ("ja", "サポートに連絡するにはどうすればよいですか？"),
        ("zh", "我如何联系客服？"),
    ],
    "two_factor": [
        ("en", "How do I enable two-factor authentication?"),
        ("es", "¿Cómo activo la autenticación de dos factores?"),
        ("hi", "मैं टू-फैक्टर प्रमाणीकरण कैसे सक्षम करूँ?"),
        ("ja", "二段階認証を有効にするには？"),
    ],
    "export_data": [
        ("en", "How do I export my data?"),
        ("ja", "データをエクスポートするにはどうすればよいですか？"),
        ("zh", "我如何导出我的数据？"),
        ("fr", "Comment exporter mes données ?"),
    ],
    "pay_bill": [
        ("en", "How do I pay my bill?"),
        ("de", "Wie bezahle ich meine Rechnung?"),
        ("es", "¿Cómo pago mi factura?"),
        ("fr", "Comment régler ma facture ?"),
    ],
    "upgrade_plan": [
        ("en", "How do I upgrade my plan?"),
        ("de", "Wie kann ich meinen Tarif upgraden?"),
        ("hi", "मैं अपना प्लान कैसे अपग्रेड करूँ?"),
    ],
    "invoice": [
        ("en", "Where can I find my invoice?"),
        ("es", "¿Dónde encuentro mi factura?"),
        ("fr", "Où trouver ma facture ?"),
    ],
    "delete_account": [
        ("en", "How do I delete my account?"),
        ("zh", "我如何删除我的账户？"),
        ("ja", "アカウントを削除するにはどうすればよいですか？"),
    ],
    "change_email": [
        ("en", "How do I change my email address?"),
        ("fr", "Comment changer mon adresse e-mail ?"),
        ("de", "Wie ändere ich meine E-Mail-Adresse?"),
    ],
    "invite_teammate": [
        ("en", "How do I invite a teammate?"),
        ("pt", "Como faço para convidar um colega?"),
        ("es", "¿Cómo invito a un compañero?"),
    ],
    "billing_address": [
        ("en", "How do I change my billing address?"),
        ("hi", "मैं अपना बिलिंग पता कैसे बदलूँ?"),
        ("pt", "Como altero meu endereço de cobrança?"),
    ],
}

_model = None


def load_model(name=None):
    """Load (and cache) the sentence-transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(name or MODEL)
    return _model


def embed(texts, model=None):
    """L2-normalised embeddings."""
    model = model if model is not None else load_model()
    return np.asarray(model.encode(texts, normalize_embeddings=True),
                      dtype=np.float64)
