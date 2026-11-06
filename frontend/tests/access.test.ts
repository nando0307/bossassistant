import assert from 'node:assert/strict'
import { test } from 'node:test'
import { hasAskAccess } from '../src/access.ts'

test('hosted access contract: discovery must settle before asking', () => {
  const employee = { persona: 'employee', token: 'employee-token' }
  const executive = { persona: 'executive', token: 'executive-token' }

  assert.equal(hasAskAccess('pending', 'employee', null), false, 'cold start')
  assert.equal(hasAskAccess('failed', 'employee', null), false, 'discovery failure')
  assert.equal(hasAskAccess('pending', 'employee', null), false, 'discovery retry')
  assert.equal(hasAskAccess('enabled', 'employee', null), false, 'token pending or failed')
  assert.equal(hasAskAccess('enabled', 'employee', employee), true, 'demo ready')
  assert.equal(hasAskAccess('enabled', 'executive', employee), false, 'identity switch')
  assert.equal(hasAskAccess('enabled', 'executive', executive), true, 'new identity ready')
  assert.equal(hasAskAccess('enabled', 'executive', null), false, '401 invalidates access')
  assert.equal(hasAskAccess('pending', 'executive', executive), false, 'retry blocks stale auth')
  assert.equal(hasAskAccess('failed', 'executive', executive), false, 'failed discovery blocks stale auth')
  assert.equal(hasAskAccess('enabled', 'employee', { persona: 'employee', token: '' }), false, 'empty token')
  assert.equal(hasAskAccess('disabled', 'employee', null), true, 'confirmed non-demo path')
})
