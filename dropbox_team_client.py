# import pandas as pd
# from io import StringIO, BytesIO
import os
from typing import Optional, Dict, List, Union, Any
from datetime import datetime, timedelta
# import json
import requests
import dropbox
from dropbox import Dropbox, DropboxTeam
# from dropbox.exceptions import ApiError, AuthError
from dropbox.files import FileMetadata, FolderMetadata, WriteMode
# from dropbox.team import TeamFolderMetadata, MemberProfile
import logging
from dotenv import load_dotenv

class TeamDropboxClientManager:
    def __init__(self, log_level: int = logging.INFO):
        # Set up logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('TeamDropboxManager')
        
        # Initialize credentials
        self._init_credentials()
        
        # Initialize state
        self.dbx_team: Optional[DropboxTeam] = None
        self.current_access_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
        self.team_space_client: Optional[Dropbox] = None
        self.admin_member_id: Optional[str] = None
        
        # Initialize the clients
        self._init_clients()

    def _init_credentials(self) -> None:
        """Initialize and validate credentials from environment variables."""
        load_dotenv()
        
        self.APP_KEY = os.getenv('DROPBOX_APP_KEY')
        self.APP_SECRET = os.getenv('DROPBOX_APP_SECRET')
        self.TEAM_REFRESH_TOKEN = os.getenv('DROPBOX_TEAM_REFRESH_TOKEN')
        
        missing = []
        if not self.APP_KEY: missing.append('DROPBOX_APP_KEY')
        if not self.APP_SECRET: missing.append('DROPBOX_APP_SECRET')
        if not self.TEAM_REFRESH_TOKEN: missing.append('DROPBOX_TEAM_REFRESH_TOKEN')
        
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")

    def _is_token_expired(self) -> bool:
        """Check if the current access token is expired or about to expire."""
        if not self.token_expiry or not self.current_access_token:
            return True
        
        # Consider token expired if it expires within the next 5 minutes
        return datetime.now() >= (self.token_expiry - timedelta(minutes=5))

    def refresh_access_token(self) -> str:
        """Refresh the team access token."""
        try:
            self.logger.info("Refreshing access token...")
            
            response = requests.post(
                "https://api.dropboxapi.com/oauth2/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self.TEAM_REFRESH_TOKEN,
                    "client_id": self.APP_KEY,
                    "client_secret": self.APP_SECRET
                },
                timeout=30  # Add timeout
            )
            
            response.raise_for_status()
            token_data = response.json()
            
            self.current_access_token = token_data['access_token']
            # Set expiry to 4 hours minus 5 minutes for safety
            self.token_expiry = datetime.now() + timedelta(hours=3, minutes=55)
            
            self.logger.info("Access token refreshed successfully")
            return self.current_access_token
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error during token refresh: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Token refresh failed: {str(e)}")
            raise

    def _ensure_valid_token(self) -> None:
        """Ensure we have a valid access token, refreshing if necessary."""
        if self._is_token_expired():
            self.logger.info("Token expired or missing, refreshing...")
            self.refresh_access_token()
            # Re-initialize clients with new token
            self._init_clients()

    def _get_admin_member_id(self) -> str:
        """Get an admin team member ID."""
        try:
            # List all team members
            result = self.dbx_team.team_members_list()
            
            # Find an active admin member
            admin_member = next(
                (member for member in result.members 
                 if member.profile.status.is_active and 
                 member.role.is_team_admin),
                None
            )
            
            if not admin_member:
                raise ValueError("No active admin team members found")
                
            self.logger.info(f"Using admin member: {admin_member.profile.email}")
            return admin_member.profile.team_member_id
            
        except Exception as e:
            self.logger.error(f"Error getting admin member ID: {str(e)}")
            raise

    def _init_clients(self) -> None:
        """Initialize team client and configure team space access."""
        try:
            # Ensure we have a valid token
            if not self.current_access_token or self._is_token_expired():
                self.refresh_access_token()
            
            self.dbx_team = DropboxTeam(self.current_access_token)
            
            # Get an admin member ID - always refresh this to ensure it's valid
            self.admin_member_id = self._get_admin_member_id()
            
            # Get an admin client
            admin_client = self.dbx_team.as_admin(self.admin_member_id)
            
            # Get the root namespace ID for team space
            root_info = admin_client.users_get_current_account().root_info
            root_namespace_id = root_info.root_namespace_id
            
            # Configure client with team space path root
            self.team_space_client = admin_client.with_path_root(
                dropbox.common.PathRoot.root(root_namespace_id)
            )
            
            self.logger.info("Successfully initialized team space client")
            
        except Exception as e:
            self.logger.error(f"Error initializing clients: {str(e)}")
            raise

    def get_folder_contents(self, path: str) -> List[Dict[str, Any]]:
        """
        List contents of a team space folder.
        
        Args:
            path: Folder path starting with /
        """
        try:
            self._ensure_valid_token()
            
            if not self.team_space_client:
                raise ValueError("Team space client not initialized")
            
            contents = []
            
            # Get initial results
            result = self.team_space_client.files_list_folder(path)
            self._process_entries(result.entries, contents)
            
            # Continue if there are more entries
            while result.has_more:
                result = self.team_space_client.files_list_folder_continue(result.cursor)
                self._process_entries(result.entries, contents)
                
            return contents
            
        except Exception as e:
            self.logger.error(f"Error listing folder {path}: {str(e)}")
            raise

    def _process_entries(self, entries, contents: List[Dict[str, Any]]) -> None:
        """Process and format folder entries."""
        for entry in entries:
            item = {
                'name': entry.name,
                'path': entry.path_display,
                'type': 'folder' if isinstance(entry, FolderMetadata) else 'file'
            }
            
            if isinstance(entry, FileMetadata):
                item.update({
                    'size': entry.size,
                    'modified': entry.server_modified.isoformat(),
                    'content_hash': entry.content_hash
                })
                
            contents.append(item)

    def get_file(self, path: str) -> Optional[bytes]:
        """
        Download a file from team space.
        
        Args:
            path: File path starting with /
        """
        try:
            self._ensure_valid_token()
            
            if not self.team_space_client:
                raise ValueError("Team space client not initialized")
                
            _, response = self.team_space_client.files_download(path)
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error downloading file {path}: {str(e)}")
            raise

    def upload_file(self, file_content: bytes, path: str, mode: str = 'overwrite') -> FileMetadata:
        """
        Upload a file to team space.
        
        Args:
            file_content: The file content as bytes
            path: Destination path starting with /
            mode: Upload mode - 'overwrite' or 'add'
        
        Returns:
            FileMetadata object with information about the uploaded file
        """
        try:
            self.logger.info(f"Starting upload to {path}")
            
            # Ensure we have a valid token before upload
            self._ensure_valid_token()
            
            if not self.team_space_client:
                raise ValueError("Team space client not initialized")
            
            # Set write mode
            write_mode = WriteMode.overwrite if mode == 'overwrite' else WriteMode.add
            
            # Validate file content
            if not isinstance(file_content, bytes):
                raise ValueError("file_content must be bytes")
            
            if len(file_content) == 0:
                raise ValueError("file_content cannot be empty")
            
            # Upload the file directly - let Dropbox handle folder creation
            try:
                result = self.team_space_client.files_upload(
                    file_content,
                    path,
                    mode=write_mode,
                    autorename=True,  # Automatically rename if file exists and mode is 'add'
                    mute=True  # Don't notify team members about the upload
                )
                
                self.logger.info(f"Successfully uploaded file to {path} ({len(file_content)} bytes)")
                return result
                
            except dropbox.exceptions.ApiError as e:
                # UploadError.get_path() returns an UploadWriteFailed
                # wrapper — the WriteError lives one level down at
                # .reason (unlike CreateFolderError.get_path(), which IS
                # the WriteError). Checking is_conflict() on the wrapper
                # raised AttributeError and masked the real upload error
                # (field report 2026-08-20).
                if (e.error.is_path()
                        and e.error.get_path().reason.is_conflict()):
                    self.logger.warning(f"File conflict at {path}, retrying with autorename")
                    # Retry with autorename enabled
                    result = self.team_space_client.files_upload(
                        file_content,
                        path,
                        mode=WriteMode.add,  # Use add mode for autorename
                        autorename=True,
                        mute=True
                    )
                    self.logger.info(f"Successfully uploaded file to {result.path_display}")
                    return result
                else:
                    raise
            
        except dropbox.exceptions.AuthError as e:
            self.logger.error(f"Authentication error during upload: {str(e)}")
            # Try to refresh token and retry once
            try:
                self.logger.info("Attempting to refresh token and retry upload...")
                self.refresh_access_token()
                self._init_clients()
                
                result = self.team_space_client.files_upload(
                    file_content,
                    path,
                    mode=write_mode,
                    autorename=True,
                    mute=True
                )
                self.logger.info(f"Successfully uploaded file after token refresh: {path}")
                return result
                
            except Exception as retry_e:
                self.logger.error(f"Retry after token refresh failed: {str(retry_e)}")
                raise
                
        except Exception as e:
            self.logger.error(f"Error uploading file to {path}: {str(e)}")
            raise

    def create_folder(self, path: str) -> Optional[FolderMetadata]:
        """
        Create a folder in team space.
        
        Args:
            path: Folder path starting with /
            
        Returns:
            FolderMetadata object with information about the created folder
        """
        try:
            self._ensure_valid_token()
            
            if not self.team_space_client:
                raise ValueError("Team space client not initialized")
                
            result = self.team_space_client.files_create_folder_v2(path)
            self.logger.info(f"Successfully created folder: {path}")
            return result.metadata
            
        except dropbox.exceptions.ApiError as e:
            # Check if folder already exists
            if e.error.is_path() and e.error.get_path().is_conflict():
                self.logger.info(f"Folder already exists: {path}")
                return None  # Folder already exists, which is fine
            else:
                self.logger.error(f"API error creating folder {path}: {str(e)}")
                raise
        except Exception as e:
            self.logger.error(f"Error creating folder {path}: {str(e)}")
            raise

    def ensure_folder_exists(self, path: str) -> None:
        """
        Ensure a folder path exists, creating parent folders if necessary.
        
        Args:
            path: Folder path starting with /
        """
        try:
            self._ensure_valid_token()
            
            # Split the path and create each level if needed
            parts = [p for p in path.split('/') if p]
            current_path = ''
            
            for part in parts:
                current_path += f'/{part}'
                try:
                    self.create_folder(current_path)
                except dropbox.exceptions.ApiError as e:
                    # Ignore if folder already exists
                    if not (e.error.is_path() and e.error.get_path().is_conflict()):
                        raise
                        
        except Exception as e:
            self.logger.error(f"Error ensuring folder exists {path}: {str(e)}")
            raise