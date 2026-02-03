# Security

## Automated Scanning

### CI Pipeline
```bash
# Runs on every push
- Bandit: Python security vulnerabilities
- Trufflehog: Secret detection
- Safety: Dependency vulnerabilities
```

### Local Scanning
```bash
# Security audit
make security

# Individual scans
bandit -r backend/ python/ -c pyproject.toml
safety check -r requirements.txt
```

## Security Middleware

### Rate Limiting
- 100 requests/minute per IP
- Prevents abuse and DoS
- Configurable in `backend/app/config.py`

### Security Headers
```python
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000
```

### Request Tracking
- Unique request ID per request
- Enables distributed tracing
- Logged in all operations

## Input Validation

### File Uploads
- Max size: 10MB
- Allowed types: JPEG, PNG
- Content-type validation
- Dimension limits: 32x32 to 4096x4096

### API Validation
- Pydantic models for all inputs
- Type checking
- Range validation
- SQL injection prevention (SQLAlchemy ORM)

## Best Practices

✅ **Do**:
- Use environment variables for secrets
- Validate all inputs
- Use HTTPS in production
- Keep dependencies updated
- Run security scans regularly

❌ **Don't**:
- Hardcode secrets
- Trust user input
- Expose internal errors
- Use default credentials
- Skip security updates

## Secrets Management

### Development
```bash
# .env file (not committed)
DATABASE_URL=sqlite:///./cv_classification.db
API_KEY=your_key_here
```

### Production
Use managed services:
- AWS Secrets Manager
- HashiCorp Vault
- Azure Key Vault

## Vulnerability Response

1. Security scan finds issue
2. Check severity (Critical/High/Medium/Low)
3. Update dependency or patch code
4. Re-run tests
5. Deploy fix
6. Document in CHANGELOG

## Security Checklist

- [x] Automated security scanning
- [x] Input validation
- [x] Rate limiting
- [x] Security headers
- [x] No hardcoded secrets
- [x] HTTPS ready
- [ ] Secrets management (production)
- [ ] WAF (production)
- [ ] DDoS protection (production)
