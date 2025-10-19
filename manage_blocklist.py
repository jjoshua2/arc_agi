#!/usr/bin/env python3
"""
Command-line utility to manage the primitive blocklist.
Useful for debugging and manual blocklist management.
"""

import argparse
import json
import time
from src.primitive_blocklist import get_primitive_blocklist

def main():
    parser = argparse.ArgumentParser(description="Manage primitive blocklist")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show blocklist statistics')
    
    # List command  
    list_parser = subparsers.add_parser('list', help='List blocked primitives')
    list_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed info')
    
    # Block command
    block_parser = subparsers.add_parser('block', help='Manually block a primitive')
    block_parser.add_argument('primitive_id', help='ID of primitive to block')
    block_parser.add_argument('--reason', default='manual', help='Reason for blocking')
    
    # Unblock command
    unblock_parser = subparsers.add_parser('unblock', help='Unblock a primitive')
    unblock_parser.add_argument('primitive_id', help='ID of primitive to unblock')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear entire blocklist')
    clear_parser.add_argument('--confirm', action='store_true', help='Confirm clearing')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export blocklist to JSON')
    export_parser.add_argument('filename', nargs='?', default='blocklist_export.json', 
                              help='Output filename')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    blocklist = get_primitive_blocklist()
    
    if args.command == 'status':
        stats = blocklist.get_statistics()
        print(f"=== Primitive Blocklist Status ===")
        print(f"Total blocked primitives: {stats['total_blocked']}")
        print(f"Total failures recorded: {stats['total_failures_recorded']}")
        print(f"Recent failures (1h): {stats['recent_failures']}")
        
        if stats['most_problematic']:
            print(f"\nMost problematic primitives:")
            for primitive_id, count in stats['most_problematic'][:10]:
                print(f"  {primitive_id}: {count} failures")
        
        if stats['blocked_primitives']:
            print(f"\nBlocked primitives: {len(stats['blocked_primitives'])}")
            for pid in stats['blocked_primitives'][:10]:
                print(f"  {pid}")
            if len(stats['blocked_primitives']) > 10:
                print(f"  ... and {len(stats['blocked_primitives']) - 10} more")
    
    elif args.command == 'list':
        stats = blocklist.get_statistics()
        if not stats['blocked_primitives']:
            print("No primitives are currently blocked.")
            return
            
        print(f"Blocked primitives ({len(stats['blocked_primitives'])}):")
        for pid in stats['blocked_primitives']:
            if args.verbose:
                failure_count = blocklist.failure_counts.get(pid, 0)
                recent_failures = len([
                    ts for ts in blocklist.failure_history.get(pid, [])
                    if time.time() - ts < blocklist.time_window
                ])
                print(f"  {pid} (total: {failure_count}, recent: {recent_failures})")
            else:
                print(f"  {pid}")
    
    elif args.command == 'block':
        # Manually record failures to trigger blocking
        for _ in range(blocklist.crash_threshold):
            newly_blocked = blocklist.record_failure(args.primitive_id, args.reason)
            if newly_blocked:
                break
        if blocklist.is_blocked(args.primitive_id):
            print(f"✅ Primitive {args.primitive_id} is now blocked")
        else:
            print(f"❌ Failed to block primitive {args.primitive_id}")
    
    elif args.command == 'unblock':
        if blocklist.unblock_primitive(args.primitive_id):
            print(f"✅ Unblocked primitive {args.primitive_id}")
        else:
            print(f"❌ Primitive {args.primitive_id} was not blocked")
    
    elif args.command == 'clear':
        if not args.confirm:
            stats = blocklist.get_statistics()
            if stats['total_blocked'] > 0:
                print(f"This will unblock {stats['total_blocked']} primitives.")
                print("Use --confirm to proceed.")
                return
        
        blocklist.clear_blocklist()
        print("✅ Blocklist cleared")
    
    elif args.command == 'export':
        stats = blocklist.get_statistics()
        
        export_data = {
            'statistics': stats,
            'failure_counts': dict(blocklist.failure_counts),
            'settings': {
                'crash_threshold': blocklist.crash_threshold,
                'time_window': blocklist.time_window
            }
        }
        
        with open(args.filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Blocklist exported to {args.filename}")

if __name__ == "__main__":
    main()