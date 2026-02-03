# Database Migrations

This project uses Alembic for database schema migrations.

## Setup

```bash
# Initialize Alembic (already done)
alembic init migrations

# Create first migration
alembic revision --autogenerate -m "Initial schema"

# Apply migrations
alembic upgrade head
```

## Creating Migrations

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "Add user table"

# Create empty migration
alembic revision -m "Custom migration"
```

## Applying Migrations

```bash
# Upgrade to latest
alembic upgrade head

# Upgrade one version
alembic upgrade +1

# Downgrade one version
alembic downgrade -1

# Downgrade to specific revision
alembic downgrade <revision_id>
```

## Rollback Procedures

### Rollback Last Migration
```bash
alembic downgrade -1
```

### Rollback to Specific Version
```bash
# Check current version
alembic current

# List all revisions
alembic history

# Rollback to specific revision
alembic downgrade <revision_id>
```

### Emergency Rollback
```bash
# Rollback all migrations
alembic downgrade base

# Reapply from scratch
alembic upgrade head
```

## Best Practices

1. **Always test migrations** in development before production
2. **Backup database** before running migrations in production
3. **Review auto-generated migrations** - they may need manual adjustments
4. **Never edit applied migrations** - create new ones instead
5. **Keep migrations small** - one logical change per migration

## Configuration

Database URL is configured in `backend/app/config.py`:
```python
database_url: str = "sqlite:///./cv_classification.db"
```

Override with environment variable:
```bash
export DATABASE_URL="postgresql://user:pass@localhost/dbname"
```
