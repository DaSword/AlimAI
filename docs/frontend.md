# Frontend Documentation

## Overview

The Alim AI frontend is a React-based single-page application built with TypeScript, providing an Islamic knowledge assistant interface. It features a chat interface for interacting with an AI assistant, admin tools for managing knowledge base collections, and a comprehensive theming system inspired by Islamic design principles.

## Architecture

### Technology Stack

- **Framework**: React 18.2.0 with TypeScript
- **Build Tool**: Vite 5.0.8
- **Routing**: React Router DOM 6.21.0
- **Styling**: Tailwind CSS 3.4.0 with custom Islamic-inspired design system
- **State Management**: React Hooks (useState, useEffect, useRef)
- **API Client**: LangGraph SDK for chat operations, Axios for admin operations
- **UI Components**: Custom component library built on class-variance-authority
- **Icons**: Lucide React

### Project Structure

```
frontend/
├── src/
│   ├── api/              # API client and types
│   │   └── client.ts     # LangGraph SDK client, admin API client, type definitions
│   ├── components/
│   │   ├── admin/        # Admin panel components
│   │   ├── chat/         # Chat interface components
│   │   └── ui/           # Reusable UI primitives
│   ├── pages/            # Page components
│   ├── App.tsx           # Root component with routing
│   ├── main.tsx          # Application entry point
│   └── index.css         # Global styles and theme definitions
├── index.html            # HTML template
├── vite.config.ts        # Vite configuration
├── tailwind.config.js    # Tailwind CSS configuration
├── tsconfig.json         # TypeScript configuration
└── package.json          # Dependencies and scripts
```

## Core Concepts

### Theme System

The application implements a comprehensive dark/light theme system with Islamic-inspired color palette. Theme preferences are stored in localStorage and synchronized across the application.

**Theme Modes:**
- `light`: Bright theme with light backgrounds
- `dark`: Dark theme optimized for low-light environments
- `system`: Automatically matches the user's system preference

**Color Palette:**
- **Primary (Emerald Green)**: Represents paradise/Jannah - `hsl(160 84% 39%)` in light mode
- **Secondary (Gold)**: Represents divine light and illumination - `hsl(45 100% 51%)`
- **Accent (Teal)**: Represents wisdom and tranquility - `hsl(173 80% 40%)`
- **Islamic Navy**: Used for borders and structural elements

Colors are defined as CSS custom properties in `index.css` and referenced throughout the application using Tailwind's HSL color system.

### Islamic Design Elements

The application incorporates several Islamic design elements:

1. **Geometric Patterns**: Background patterns created using CSS gradients and radial patterns
2. **Arabic Typography**: Support for Arabic text with RTL (right-to-left) direction
3. **Decorative Elements**: Custom SVG ornaments and dividers inspired by Islamic art
4. **Elegant Fonts**: Google Fonts integration including:
   - Inter (sans-serif) for UI elements
   - Amiri (serif) for Arabic text
   - Scheherazade New for decorative Arabic text
   - Playfair Display for elegant headings

### Routing

The application uses React Router DOM with a single route configuration:
- `/` - Chat page (default route)

The routing structure is minimal and can be easily extended to add additional pages. All routes are defined in `App.tsx`.

## Component Architecture

### Entry Point

**`main.tsx`**: 
- Initializes React application
- Mounts `App` component to the DOM
- Enables React Strict Mode for development

**`App.tsx`**:
- Sets up BrowserRouter for navigation
- Initializes theme system on mount
- Loads saved theme preference from localStorage
- Applies theme to document root element
- Defines route structure

### Chat Page (`pages/Chat.tsx`)

The main application page that orchestrates the chat interface. This component manages:

**State Management:**
- Message history (array of Message objects)
- Current input value
- Loading states during API calls
- Thread ID for LangGraph conversation threads
- Streaming message content during real-time responses
- Chat thread list for conversation history
- Sidebar visibility state
- Modal visibility states (settings, admin)
- Theme mode preference
- User profile information (full name, email)
- Editing states for profile fields

**Key Functionality:**
- Thread initialization: Creates a new LangGraph thread on component mount
- Message sending: Handles user input, sends to API, processes streaming responses
- Message state management: Accumulates streaming content and extracts sources
- Thread management: Saves conversations when starting new chats
- Theme application: Synchronizes theme changes with localStorage and DOM
- Auto-scrolling: Automatically scrolls to bottom when new messages arrive

**Component Composition:**
- `ChatSidebar`: Conversation history and user menu
- `WelcomeScreen`: Initial empty state with quick action prompts
- `MessageBubble`: Individual message display with source citations
- `ChatInput`: Text input with auto-resizing textarea
- `SettingsModal`: User preferences and theme selection
- `AdminModal`: Administrative tools access

### Chat Components (`components/chat/`)

#### `ChatSidebar.tsx`

Sidebar component providing navigation and user controls.

**Features:**
- Collapsible sidebar with smooth width transitions
- Logo and branding display
- "New chat" button to start fresh conversations
- Recent conversations list (currently displays but doesn't implement thread switching)
- User menu dropdown with:
  - Settings access
  - Admin panel access
  - Logout functionality (placeholder)

**Implementation Details:**
- Uses `min-w-64` class to maintain minimum width during collapse animation
- Click-outside detection for closing user menu dropdown
- Responsive design with overflow handling for long thread titles

#### `ChatInput.tsx`

Text input component for composing messages.

**Features:**
- Auto-resizing textarea (grows up to 200px height)
- Enter key to send (Shift+Enter for new line)
- Loading state disables input during API calls
- Send button with loading spinner
- Islamic gradient styling on send button

**Technical Details:**
- Uses ref to programmatically adjust textarea height
- Minimum height of 48px for accessibility
- Backdrop blur effect on container for visual separation

#### `MessageBubble.tsx`

Displays individual chat messages with formatting and source citations.

**Features:**
- Distinct styling for user vs assistant messages
- User messages: Right-aligned with primary color background
- Assistant messages: Left-aligned with avatar icon and content area
- Source citations display when available:
  - Book title and reference
  - Excerpt preview (2 lines max)
  - Islamic card border styling
- Streaming indicator: Pulsing cursor during real-time responses

**Message Structure:**
Each message extends the base `ChatMessage` interface with:
- Unique ID
- Timestamp
- Optional sources array with book_title, reference, and text_content

#### `WelcomeScreen.tsx`

Initial screen shown when no messages exist.

**Features:**
- Decorative Bismillah (Arabic invocation) with gradient text effect
- Animated logo with glow effects
- Quick action cards for common queries:
  - Quran topics
  - Hadith requests
  - Prayer guidance
  - Fiqh questions
- Corner ornaments for visual enhancement
- Islamic dividers for section separation

**Interaction:**
- Clicking quick action cards populates the input field with the prompt text

#### `SettingsModal.tsx`

Modal dialog for user preferences and account settings.

**Features:**
- Profile section:
  - Full name editing (inline edit with pencil icon)
  - Email editing (inline edit with pencil icon)
  - Read-only mode with click-to-edit functionality
- Appearance section:
  - Theme selection cards (Light, System, Dark)
  - Visual theme preview with icons
  - Active theme indicator

**State Management:**
- Theme changes are immediately applied and persisted
- Editing states control input field behavior
- Modal closes on backdrop click or X button

#### `AdminModal.tsx`

Container modal for administrative tools.

**Features:**
- Tabbed interface with three sections:
  - Ingestion: File upload and processing
  - Collections: Vector database management
  - Models: LLM model status and management
- Delegates to specialized admin components

#### `IslamicDecorations.tsx`

Reusable SVG components for Islamic design elements.

**Components:**
- `IslamicBookIcon`: SVG icon representing a book with geometric patterns
- `CornerOrnament`: Decorative corner element (supports 4 positions: tl, tr, bl, br)
- `IslamicDivider`: Decorative horizontal divider with wave pattern and ornaments

**Design Philosophy:**
- Uses currentColor for theme compatibility
- Opacity variations for subtle effects
- Responsive to dark/light theme changes

#### `index.ts`

Barrel export file for chat components, providing centralized imports.

### Admin Components (`components/admin/`)

#### `IngestionPanel.tsx`

Interface for uploading and ingesting Islamic text data.

**Features:**
- File upload with JSON file type restriction
- Source type selection dropdown:
  - Quran
  - Hadith
  - Tafsir
  - Fiqh
  - Seerah
  - Aqidah
- Upload progress tracking with status messages
- Visual feedback for upload states:
  - Uploading: Blue background with spinner
  - Completed: Green background with checkmark
  - Error: Red background with error icon
- File format guidelines display

**API Integration:**
- Uses `uploadAndIngestFile` function from API client
- Handles async file upload with FormData
- Processes response status and progress updates

#### `CollectionManager.tsx`

Manages Qdrant vector database collections.

**Features:**
- Collection list display with:
  - Collection name
  - Point count
  - Vector count
  - Vector size configuration
  - Distance metric configuration
- Actions per collection:
  - Clear: Removes all data from collection (with confirmation)
  - Delete: Permanently removes collection (with confirmation)
- Refresh button to reload collection list
- Loading states during operations

**API Integration:**
- `listCollections`: Fetches all collections
- `clearCollection`: Removes all points from a collection
- `deleteCollection`: Deletes the collection entirely

**Error Handling:**
- Confirmation dialogs prevent accidental deletions
- Alert messages for failed operations
- Loading indicators prevent duplicate actions

#### `ModelStatus.tsx`

Displays system health and LLM model information.

**Features:**
- System health dashboard showing status of:
  - Ollama (port 11434)
  - LM Studio (port 1234)
  - Qdrant (ports 6333 HTTP, 6334 gRPC)
  - LangGraph server
- Model list with:
  - Model name
  - Model size (in GB)
  - Loaded status indicator
  - Pull button for unloaded models
- Service information panel with:
  - Connection details
  - Recommended models
  - Command examples

**API Integration:**
- `checkHealth`: Fetches backend service status
- `listModels`: Retrieves available Ollama models
- `pullModel`: Downloads model from Ollama registry

**Visual Indicators:**
- Green checkmark for healthy services
- Red X for offline services
- Status badges showing ports and connection info

### UI Components (`components/ui/`)

These are reusable primitive components built following a design system pattern.

#### `button.tsx`

Button component with variant and size options.

**Variants:**
- `default`: Primary action with Islamic emerald background
- `destructive`: Delete/danger actions with red background
- `outline`: Secondary actions with border
- `secondary`: Alternative primary with gold background
- `ghost`: Minimal styling for icon buttons
- `link`: Text link style

**Sizes:**
- `default`: Standard height (h-10)
- `sm`: Small (h-9)
- `lg`: Large (h-11)
- `icon`: Square icon button (h-10 w-10)

**Implementation:**
- Uses `class-variance-authority` for variant management
- Merges classes with `cn` utility function
- Supports all standard button HTML attributes

#### `card.tsx`

Card container component with semantic sub-components.

**Components:**
- `Card`: Main container with border and shadow
- `CardHeader`: Header section with spacing
- `CardTitle`: Large title text
- `CardDescription`: Muted description text
- `CardContent`: Main content area
- `CardFooter`: Footer section

**Usage Pattern:**
Cards are composed hierarchically for structured content display.

#### `input.tsx`

Text input component with consistent styling.

**Features:**
- File input support
- Placeholder styling
- Focus ring with primary color
- Disabled state styling
- Responsive text sizing

#### `textarea.tsx`

Multi-line text input component.

**Features:**
- Similar styling to input component
- Resizable (controlled by parent components)
- Consistent focus and disabled states

#### `tabs.tsx`

Tabbed interface component.

**Features:**
- Context-based state management
- Multiple tab panels
- Active tab highlighting
- Keyboard navigation support

**Usage:**
Used in AdminModal for organizing admin tools into sections.

### API Client (`api/client.ts`)

Centralized API communication layer.

#### Configuration

**LangGraph Client:**
- Uses `@langchain/langgraph-sdk` for chat operations
- Base URL from environment variable `VITE_LANGGRAPH_URL` (defaults to `http://localhost:8123`)
- Graph name: `rag_assistant` (must match backend configuration)

**Admin API Client:**
- Axios instance for REST operations
- Base URL: `${LANGGRAPH_URL}/api/admin`
- JSON content type headers

#### Chat API Functions

**`createThread()`:**
- Creates a new conversation thread
- Returns thread ID string
- Used to initialize new conversations

**`streamChatResponse(threadId, message)`:**
- Streams chat responses from LangGraph
- Returns async generator yielding StreamEvent objects
- Handles LangGraph event format:
  - `event: "values"` contains message data and sources
  - Extracts assistant messages and source citations
- Used for real-time response streaming

**`getThreadHistory(threadId)`:**
- Retrieves full conversation history for a thread
- Returns array of ChatMessage objects
- Currently available but not actively used in UI

#### Admin API Functions

**Ingestion:**
- `uploadAndIngestFile(file, sourceType)`: Uploads JSON file and starts ingestion
- `getIngestionStatus(taskId)`: Checks status of ingestion task

**Collections:**
- `listCollections()`: Gets all Qdrant collections
- `getCollectionInfo(collectionName)`: Gets detailed collection info
- `deleteCollection(collectionName)`: Permanently deletes collection
- `clearCollection(collectionName)`: Removes all data from collection

**Models:**
- `listModels()`: Lists available Ollama models
- `getModelStatus(modelName)`: Gets model loading status
- `pullModel(modelName)`: Downloads model from Ollama

**Health:**
- `checkHealth()`: Checks backend service status

#### Type Definitions

**ChatMessage:**
```typescript
interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}
```

**StreamEvent:**
```typescript
interface StreamEvent {
  event: string;
  data: any;
}
```

**CollectionInfo:**
Contains collection metadata including name, point count, vector count, and configuration.

**ModelInfo:**
Contains model name, size in bytes, and loaded status.

**IngestionProgress:**
Status tracking for file ingestion operations.

## Styling System

### Tailwind Configuration

**Custom Colors:**
- Islamic color palette: emerald, gold, teal with 50-900 scale
- CSS variable-based colors for theme system
- Semantic color tokens: primary, secondary, accent, destructive, muted

**Custom Utilities:**
- Islamic pattern background class
- Custom scrollbar styling
- Hide scrollbar utility
- Smooth transitions
- Fade-in animations

**Font Configuration:**
- Arabic font family: Amiri, serif
- Sans-serif: Inter
- Custom heading font: Playfair Display (via CSS class)

### CSS Architecture (`index.css`)

**CSS Layers:**
1. **Base Layer**: CSS custom properties for theme variables
2. **Components Layer**: Component-specific styles
3. **Utilities Layer**: Utility classes and animations

**Theme Variables:**
All colors defined as HSL values in CSS custom properties, enabling runtime theme switching without rebuilds.

**Islamic Design Patterns:**
- Geometric background patterns using multiple radial gradients
- Subtle dot patterns for texture
- Repeating linear gradients for structure

**Custom Utilities:**
- `.islamic-pattern`: Complex background pattern combining multiple gradients
- `.islamic-card-border`: Decorative top border with gradient
- `.islamic-gradient`: Linear gradient using emerald and teal
- `.gold-accent`: Gold color text utility
- `.decorative-corner`: Corner decoration pseudo-element
- `.heading-elegant`: Serif font for headings
- `.arabic`: RTL text direction and Arabic font family
- `.custom-scrollbar`: Styled scrollbar for chat area
- `.animate-fadeIn`: Fade-in animation keyframe

**Responsive Design:**
- Mobile-first approach
- Breakpoints handled by Tailwind defaults
- Responsive typography scaling
- Adaptive layout for sidebar and chat area

## State Management

### Component-Level State

The application uses React hooks for state management:

- **useState**: Local component state
- **useEffect**: Side effects and lifecycle management
- **useRef**: DOM references and mutable values

### Global State

Currently, there is no global state management library. Shared state is managed through:
- **localStorage**: Theme preferences and user settings
- **Props drilling**: Data passed through component hierarchy
- **API refetching**: Components fetch their own data

### State Flow

**Chat Page State:**
- Top-level state in `Chat.tsx`
- Passed down to child components via props
- Callbacks used for child-to-parent communication

**Theme State:**
- Initialized in `App.tsx` from localStorage
- Managed in `Chat.tsx` for user changes
- Applied globally via DOM class manipulation

**Thread State:**
- Thread ID stored in Chat component
- Used for all API calls
- Reset when starting new conversation

## Environment Configuration

### Environment Variables

**`VITE_LANGGRAPH_URL`**:
- Default: `http://localhost:8123`
- Override: Set in `.env` file or environment
- Used for: Backend API connection

### Build Configuration

**Vite Configuration (`vite.config.ts`):**
- React plugin for JSX transformation
- Path alias `@` for `./src` directory
- Dependency optimization for LangGraph SDK and Axios

**TypeScript Configuration:**
- Multiple config files for different contexts:
  - `tsconfig.json`: Base configuration with path mappings
  - `tsconfig.app.json`: Application-specific settings
  - `tsconfig.node.json`: Node/Vite-specific settings
- Path aliases configured for clean imports

## Development Workflow

### Available Scripts

**`npm run dev`**: Starts Vite development server with hot module replacement

**`npm run build`**: Type checks TypeScript, then builds production bundle

**`npm run preview`**: Serves production build locally for testing

**`npm run lint`**: Runs ESLint with TypeScript support

### Development Guidelines

**Component Structure:**
- Functional components with TypeScript interfaces
- Props interfaces exported for reusability
- Forward refs used for DOM components

**Code Organization:**
- One component per file
- Barrel exports for grouped components
- Type definitions colocated with implementations

**Styling Conventions:**
- Tailwind utility classes preferred
- Custom CSS classes for complex patterns
- CSS variables for theme values
- Responsive design with mobile-first approach

## Extending the Application

### Adding New Routes

1. Create page component in `src/pages/`
2. Import in `App.tsx`
3. Add Route in Routes component
4. Update navigation if needed

### Adding New Chat Components

1. Create component in `src/components/chat/`
2. Export from `src/components/chat/index.ts`
3. Import and use in `Chat.tsx`
4. Follow existing prop patterns

### Adding New Admin Features

1. Create component in `src/components/admin/`
2. Add API functions to `api/client.ts`
3. Add tab or section in `AdminModal.tsx`
4. Implement CRUD operations following existing patterns

### Modifying Theme

**Adding New Colors:**
1. Add CSS custom property in `index.css` (`:root` and `.dark`)
2. Add Tailwind color in `tailwind.config.js`
3. Use in components via Tailwind classes

**Adding New Theme Modes:**
1. Extend `ThemeMode` type in `SettingsModal.tsx`
2. Add theme card in SettingsModal
3. Update `applyTheme` function in `Chat.tsx`
4. Add CSS custom property set in `index.css`

### Integrating New API Endpoints

1. Add function to `api/client.ts`
2. Define TypeScript interfaces for request/response
3. Use axios for REST or LangGraph SDK for chat operations
4. Handle errors appropriately
5. Add loading states in consuming components

### Adding Authentication

Currently, authentication is not implemented. To add:

1. Create auth context/provider
2. Add login/logout components
3. Store tokens in localStorage or secure cookies
4. Add auth headers to API client
5. Protect routes with auth checks
6. Implement logout in ChatSidebar

### Adding Conversation History

The UI displays thread list but doesn't implement thread switching:

1. Store thread list in localStorage or backend
2. Implement thread loading in Chat component
3. Add thread selection handler in ChatSidebar
4. Update message display when switching threads
5. Add thread deletion functionality

### Performance Optimization

**Current Optimizations:**
- React.memo not used (consider for expensive components)
- useMemo/useCallback not used (consider for expensive computations)
- Code splitting not implemented (consider for admin components)

**Potential Improvements:**
- Lazy load admin components
- Memoize message list rendering
- Virtualize long conversation lists
- Optimize re-renders with React.memo

### Accessibility

**Current State:**
- Semantic HTML elements used
- Keyboard navigation in inputs
- Focus states visible
- Color contrast meets WCAG standards

**Recommendations:**
- Add ARIA labels to icon buttons
- Implement keyboard shortcuts
- Add skip navigation links
- Improve screen reader support for dynamic content

## Deployment

### Build Process

1. Run `npm run build` to create production bundle
2. Output directory: `dist/`
3. Serve static files with any web server

### Environment Setup

**Production:**
- Set `VITE_LANGGRAPH_URL` to production backend URL
- Ensure backend CORS allows frontend origin
- Configure CDN for static assets if needed

**Docker Deployment:**
- Use nginx or similar for serving static files
- Configure reverse proxy for API calls
- Set environment variables appropriately

### Assets

**Static Assets:**
- Logo: `/logo.png` (must be in public directory)
- Fonts: Loaded from Google Fonts CDN
- No additional static assets currently

**Asset Optimization:**
- Vite handles asset optimization automatically
- Images should be optimized before adding
- Consider lazy loading for images

## Troubleshooting

### Common Issues

**Theme Not Applying:**
- Check localStorage for saved theme
- Verify CSS custom properties are loaded
- Check dark class on document.documentElement

**API Connection Errors:**
- Verify `VITE_LANGGRAPH_URL` is correct
- Check backend server is running
- Verify CORS configuration on backend

**Components Not Rendering:**
- Check browser console for errors
- Verify imports are correct
- Check TypeScript errors with `npm run build`

**Styling Issues:**
- Verify Tailwind is processing classes
- Check CSS is loaded in index.html
- Verify custom CSS classes are defined

**Thread Creation Fails:**
- Check LangGraph server is running
- Verify graph name matches backend
- Check network tab for API errors

## Type Safety

The application uses TypeScript throughout for type safety:

- All components have TypeScript interfaces
- API functions have typed parameters and returns
- Props are strictly typed
- Event handlers use proper React types

**Missing Types:**
- `@/lib/utils` file is referenced but doesn't exist (should contain `cn` utility function)
- Some API responses use `any` type (should be properly typed)

**Type Definition Locations:**
- Component props: Defined in component files
- API types: Defined in `api/client.ts`
- Shared types: Can be moved to `types/` directory if needed

## Testing Considerations

Currently, no testing framework is configured. To add testing:

**Recommended:**
- Vitest for unit tests
- React Testing Library for component tests
- Playwright or Cypress for E2E tests

**Key Areas to Test:**
- API client functions
- Theme switching logic
- Message sending and receiving
- Admin operations
- Component rendering

## Dependencies

### Production Dependencies

- **react/react-dom**: UI framework
- **react-router-dom**: Client-side routing
- **lucide-react**: Icon library
- **clsx**: Conditional class names utility
- **class-variance-authority**: Component variant management
- **tailwind-merge**: Tailwind class merging utility
- **@langchain/langgraph-sdk**: LangGraph API client
- **axios**: HTTP client for admin API

### Development Dependencies

- **@vitejs/plugin-react**: Vite React plugin
- **vite**: Build tool
- **typescript**: Type checking
- **tailwindcss**: CSS framework
- **postcss/autoprefixer**: CSS processing
- **eslint**: Code linting
- **@typescript-eslint/**: TypeScript ESLint plugins

### Font Dependencies

Fonts loaded from Google Fonts CDN:
- Inter (sans-serif)
- Amiri (Arabic serif)
- Scheherazade New (Arabic decorative)
- Playfair Display (elegant serif)

## Future Considerations

### Potential Enhancements

1. **State Management**: Consider Zustand or Redux for complex state
2. **Error Boundaries**: Add React error boundaries for better error handling
3. **Toast Notifications**: Replace alerts with toast notifications
4. **Pagination**: Add pagination for long conversation lists
5. **Search**: Add search functionality for conversation history
6. **Export**: Allow exporting conversations
7. **Voice Input**: Add voice input capability
8. **Multi-language**: Expand Arabic language support
9. **Offline Support**: Add service worker for offline functionality
10. **Analytics**: Add user analytics tracking

### Technical Debt

1. Create `lib/utils.ts` file with `cn` utility function
2. Implement thread switching functionality
3. Add proper error boundaries
4. Replace `any` types with proper interfaces
5. Add loading skeletons for better UX
6. Implement proper authentication flow
7. Add proper form validation
8. Create shared constants file for magic strings

## Conclusion

This frontend application provides a solid foundation for an Islamic knowledge assistant interface. The architecture is modular, the styling system is comprehensive, and the component structure supports easy extension. Following the patterns and guidelines outlined in this documentation will ensure consistency and maintainability as the application grows.

