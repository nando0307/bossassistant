export type DiscoveryState = 'pending' | 'failed' | 'disabled' | 'enabled'

export function hasAskAccess(
  discovery: DiscoveryState,
  persona: string,
  auth: { persona: string; token: string } | null,
): boolean {
  return discovery === 'disabled' || (
    discovery === 'enabled' && auth?.persona === persona && Boolean(auth.token)
  )
}
