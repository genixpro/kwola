import React from 'react';
import Contact from './singleContact';
import HelperText from '../utility/helper-text';
import { ContactGroupItem, ContactGroupViews } from './contact.style';

export default function({ contactGroup, setSelectedContact }) {
  const ContactGroupView = ({ contacts, initChar }) => (
    <ContactGroupViews>
      <h1 className="alphabet">{initChar}</h1>
      <ContactGroupItem>
        {contacts.map(contact => (
          <Contact
            contact={contact}
            key={contact.id}
            setSelectedContact={setSelectedContact}
          />
        ))}
      </ContactGroupItem>
    </ContactGroupViews>
  );
  if (!contactGroup) {
    return <HelperText text="No Contact to View" />;
  }
  return (
    <div>
      {Object.keys(contactGroup).map(key => (
        <ContactGroupView
          contacts={contactGroup[key]}
          key={key}
          initChar={key}
        />
      ))}
    </div>
  );
}
