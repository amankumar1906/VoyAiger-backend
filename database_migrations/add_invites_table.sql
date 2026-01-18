-- Migration: Add itinerary invites functionality
-- This migration adds support for inviting users to collaborate on itineraries

-- Create the itinerary_invites table
CREATE TABLE IF NOT EXISTS public.itinerary_invites (
  id uuid NOT NULL DEFAULT uuid_generate_v4(),
  itinerary_id uuid NOT NULL,
  invited_by_user_id uuid NOT NULL,
  invitee_email character varying(255) NOT NULL,
  invitee_user_id uuid,
  status character varying(20) NOT NULL,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT itinerary_invites_pkey PRIMARY KEY (id),
  CONSTRAINT itinerary_invites_itinerary_id_fkey FOREIGN KEY (itinerary_id) REFERENCES public.itineraries(id) ON DELETE CASCADE,
  CONSTRAINT itinerary_invites_invited_by_user_id_fkey FOREIGN KEY (invited_by_user_id) REFERENCES public.users(id),
  CONSTRAINT itinerary_invites_invitee_user_id_fkey FOREIGN KEY (invitee_user_id) REFERENCES public.users(id),
  CONSTRAINT itinerary_invites_status_check CHECK (status IN ('pending', 'accepted', 'rejected')),
  CONSTRAINT itinerary_invites_unique_invite UNIQUE (itinerary_id, invitee_email)
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_invites_itinerary ON public.itinerary_invites(itinerary_id);
CREATE INDEX IF NOT EXISTS idx_invites_email ON public.itinerary_invites(invitee_email);
CREATE INDEX IF NOT EXISTS idx_invites_user_id ON public.itinerary_invites(invitee_user_id);
CREATE INDEX IF NOT EXISTS idx_invites_status ON public.itinerary_invites(status);

-- Create trigger to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to itinerary_invites table
DROP TRIGGER IF EXISTS update_itinerary_invites_updated_at ON public.itinerary_invites;
CREATE TRIGGER update_itinerary_invites_updated_at
    BEFORE UPDATE ON public.itinerary_invites
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments for documentation
COMMENT ON TABLE public.itinerary_invites IS 'Stores invitations to collaborate on itineraries';
COMMENT ON COLUMN public.itinerary_invites.invitee_email IS 'Email of the invited user (may not have account yet)';
COMMENT ON COLUMN public.itinerary_invites.invitee_user_id IS 'Linked to user account when they accept the invite';
COMMENT ON COLUMN public.itinerary_invites.status IS 'Invitation status: pending, accepted, or rejected';

-- ============================================
-- CONCURRENCY CONTROL
-- Add version tracking to itineraries table
-- ============================================

-- Add version column for optimistic locking
ALTER TABLE public.itineraries
ADD COLUMN IF NOT EXISTS version integer NOT NULL DEFAULT 1;

-- Add last_modified_by to track who made the last edit
ALTER TABLE public.itineraries
ADD COLUMN IF NOT EXISTS last_modified_by uuid;

-- Add foreign key constraint
ALTER TABLE public.itineraries
ADD CONSTRAINT itineraries_last_modified_by_fkey
FOREIGN KEY (last_modified_by) REFERENCES public.users(id);

-- Create trigger to auto-increment version on update
CREATE OR REPLACE FUNCTION increment_itinerary_version()
RETURNS TRIGGER AS $$
BEGIN
    NEW.version = OLD.version + 1;
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS increment_version_on_update ON public.itineraries;
CREATE TRIGGER increment_version_on_update
    BEFORE UPDATE ON public.itineraries
    FOR EACH ROW
    EXECUTE FUNCTION increment_itinerary_version();

COMMENT ON COLUMN public.itineraries.version IS 'Version number for optimistic locking (auto-increments on update)';
COMMENT ON COLUMN public.itineraries.last_modified_by IS 'User ID of who last modified this itinerary';
