from immobiliare.scraping.file_fetcher import FileFetcher


def main():
    fetcher = FileFetcher()
    pages = fetcher.fetch(limit=20)

    print(f"{len(pages)} pagine caricate.")
    print(pages[0]["filename"])


if __name__ == '__main__':
    main()